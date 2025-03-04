# filepath: /home/steven/code/llama_compression/train.py
import os
import glob
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
import wandb

# Accelerate & DeepSpeed
from accelerate import Accelerator, DeepSpeedPlugin

from main import SafeLlamaPixelAR

# Allow loading partially corrupted / truncated images if possible
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- 增强数据集类 ----------
class RobustImageDataset(Dataset):
    def __init__(self, root_dir, size=256, patch_size=16):
        super().__init__()
        self.root_dir = root_dir
        self.size = size
        self.patch_size = patch_size
        self._init_paths()
        
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 255)
        ])

    def _init_paths(self):
        self.paths = sorted(glob.glob(os.path.join(self.root_dir, "*.jpg")))
        print(f"Found {len(self.paths)} image files")

    def __len__(self):
        return len(self.paths)

    def random_patchify(self, img_tensor, num_patches=1):
        import random
        c, h, w = img_tensor.shape
        patches = []
        for _ in range(num_patches):
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            patch = img_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
            patches.append(patch)
        return torch.stack(patches, dim=0)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img)
            # 随机采样 1 个 16×16 patches
            patches = self.random_patchify(img_tensor, num_patches=1)
            return patches, img_tensor
        except Exception as e:
            print(f"Error loading {path}: {str(e)}, skipping...")
            return self[(idx + 1) % len(self)]

# ---------- 训练系统 ----------
class TrainingSystem:
    def __init__(self):
        # 1) Initialize wandb inside your code
        #    The accelerator can also manage wandb tracking
        wandb.init(entity="feiyangwu0309-sjtu-icec",
                   project="llama-compression", 
                   name="my-training-run")

        # 2) Create an Accelerator with DeepSpeed plugin
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            log_with="wandb",
            deepspeed_plugin=DeepSpeedPlugin(
                zero_stage=2,  # or 3 if you have enough GPU memory
                offload_optimizer_device="none"
            ),
        )
        self.accelerator.init_trackers("llama-compression")

        # 训练参数
        self.root_dir = "/remote-home/wufeiyang/dataset/flickr30k/Images"
        self.epochs = 10
        self.lr = 1e-4
        self.size = 256
        self.patch_size = 16
        self.batch_size_per_gpu = 1
        self.checkpoint_path = "training_checkpoint.pth"
        
        # 初始化组件
        self._init_dataset()
        self._init_model()
        self._init_optimizer()

        # 加载到 Accelerator
        (
            self.model,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)

        self.start_epoch = 0
        self.best_loss = float("inf")
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            self.model.module.load_state_dict(checkpoint["model_state"])  # use .module in DeepSpeed/Accelerate
            # 移除这行：self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_loss = checkpoint["best_loss"]
            print(f"Loaded checkpoint from epoch {self.start_epoch}, loss {self.best_loss:.4f}")

    def _init_dataset(self):
        dataset = RobustImageDataset(
            self.root_dir,
            size=self.size,
            patch_size=self.patch_size
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size_per_gpu,  # Accelerator handles parallel
            shuffle=True,
            num_workers=2,
            drop_last=True
        )

    def _init_model(self):
        # 建立基础模型
        base_model = SafeLlamaPixelAR(
            model_path="/remote-home/wufeiyang/saved_model"
        )
        self.model = base_model

    def _init_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=0.01
        )

    def _save_checkpoint(self, epoch, avg_loss):
        if self.accelerator.is_main_process:
            checkpoint = {
                "epoch": epoch,
                "model_state": self.model.module.state_dict(),  # use .module
                "optimizer_state": self.optimizer.state_dict(),
                "best_loss": min(avg_loss, self.best_loss),
            }
            torch.save(checkpoint, self.checkpoint_path)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        # Use tqdm on the main process only
        progress_bar = self.accelerator.prepare(
            tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
        )

        for batch_idx, (patches, _) in enumerate(progress_bar):
            b, n, c, ph, pw = patches.shape
            input_patches = patches.view(b*n, c, ph, pw)

            with self.accelerator.autocast():
                _, ce_loss = self.model(input_patches, None)
                ce_loss = ce_loss.mean()

            self.optimizer.zero_grad()
            self.accelerator.backward(ce_loss)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += ce_loss.item()
            current_loss = total_loss / (batch_idx + 1)
            if self.accelerator.is_main_process:
                progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
                wandb.log({"train_loss": current_loss})

        return total_loss / len(self.dataloader)

    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            avg_loss = self.train_epoch(epoch)
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
                wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})
                self._save_checkpoint(epoch, avg_loss)
                self.best_loss = min(avg_loss, self.best_loss)

        # 结束后保存最终模型（仅主进程）
        if self.accelerator.is_main_process:
            torch.save(self.model.module.state_dict(), "final_model.pth")
            wandb.save("final_model.pth")
            print("Training complete.")

if __name__ == "__main__":
    # 使大规模训练时的显存分配更灵活
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    system = TrainingSystem()
    system.run()