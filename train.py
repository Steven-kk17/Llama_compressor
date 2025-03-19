import os
import glob
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
import wandb
import time
import argparse

# Accelerate & DeepSpeed
from accelerate import Accelerator, DeepSpeedPlugin

from main import SafeLlamaPixelAR

# Allow loading partially corrupted / truncated images if possible
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="训练LLaMA图像压缩模型")
    parser.add_argument("--epochs", type=int, default=80,
                        help="训练的epoch数")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="每个GPU的批处理大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--dataset", type=str, 
                        default="/remote-home/wufeiyang/dataset/flickr30k/Images",
                        help="训练数据集路径")
    parser.add_argument("--model_path", type=str,
                        default="/remote-home/wufeiyang/saved_model",
                        help="基础模型路径")
    parser.add_argument("--wandb_name", type=str,
                        default=None,
                        help="wandb运行名称，不指定则自动生成")
    return parser.parse_args()

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
    def __init__(self, args):
        # 先创建Accelerator（不包含wandb配置）
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            deepspeed_plugin=DeepSpeedPlugin(
                zero_stage=2,
                offload_optimizer_device="none"
            ),
        )
        
        # 训练参数
        self.root_dir = args.dataset
        self.epochs = args.epochs
        self.lr = args.lr
        self.size = 256
        self.patch_size = 16
        self.batch_size_per_gpu = args.batch_size
        self.checkpoint_path = "training_checkpoint.pth"
        self.model_path = args.model_path
        
        # 全局步数，用于wandb记录
        self.global_step = 0
        
        # 只在主进程初始化wandb
        if self.accelerator.is_main_process:
            # 创建一个运行名称，使用时间戳或用户提供的名称
            run_name = args.wandb_name if args.wandb_name else f"llama-train-{int(time.time())}"
            
            wandb.init(
                entity="feiyangwu0309-sjtu-icec",
                project="llama-compression", 
                name=run_name,
                config={
                    "epochs": self.epochs,
                    "batch_size": self.batch_size_per_gpu,
                    "learning_rate": self.lr,
                    "model": "SafeLlamaPixelAR",
                    "patch_size": self.patch_size,
                    "image_size": self.size
                }
            )
            print(f"Wandb initialized with run name: {run_name}")
        
        # 初始化组件
        self._init_dataset()
        self._init_model()
        self._init_optimizer()

        # 加载到Accelerator
        (
            self.model,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer, 
            self.dataloader
        )

        self.start_epoch = 0
        self.best_loss = float("inf")
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            
            # 创建一个新的state_dict来匹配模型结构
            model_state = checkpoint["model_state"]
            new_state_dict = {}
            
            # 重命名键以匹配当前模型结构
            for key, value in model_state.items():
                if key.startswith("llama.model."):
                    # 将 "llama.model." 替换为 "llama."
                    new_key = key.replace("llama.model.", "llama.")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
                    
            # 加载修改后的状态字典
            self.model.module.load_state_dict(new_state_dict, strict=False)
            
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_loss = checkpoint["best_loss"]
            # 恢复global_step如果存在
            if "global_step" in checkpoint:
                self.global_step = checkpoint["global_step"]
                
            print(f"Loaded checkpoint from epoch {self.start_epoch}, loss {self.best_loss:.4f}")

    def _init_dataset(self):
        dataset = RobustImageDataset(
            self.root_dir,
            size=self.size,
            patch_size=self.patch_size
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size_per_gpu,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )

    def _init_model(self):
        # 建立基础模型
        base_model = SafeLlamaPixelAR(
            model_path=self.model_path
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
                "model_state": self.model.module.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_loss": min(avg_loss, self.best_loss),
                "global_step": self.global_step,
            }
            torch.save(checkpoint, self.checkpoint_path)
            
            # 每10个epoch保存一个模型版本
            if (epoch + 1) % 10 == 0:
                epoch_path = f"model_epoch_{epoch+1}.pth"
                torch.save(self.model.module.state_dict(), epoch_path)
                print(f"Saved model at epoch {epoch+1} to {epoch_path}")
                if wandb.run is not None:
                    wandb.save(epoch_path)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        log_interval = 50  # 每50个batch记录一次wandb（降低API调用频率）

        # 只在主进程上使用tqdm
        dataloader_iterator = self.dataloader
        if self.accelerator.is_main_process:
            dataloader_iterator = tqdm(
                self.dataloader, 
                desc=f"Epoch {epoch+1}/{self.epochs}"
            )

        for batch_idx, (patches, _) in enumerate(dataloader_iterator):
            # 增加全局步数
            self.global_step += 1
            
            b, n, c, ph, pw = patches.shape
            input_patches = patches.view(b*n, c, ph, pw)

            with self.accelerator.autocast():
                _, ce_loss = self.model(input_patches, None)
                ce_loss = ce_loss.mean()

            self.optimizer.zero_grad()
            self.accelerator.backward(ce_loss)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 收集loss
            total_loss += ce_loss.item()
            current_loss = total_loss / (batch_idx + 1)
            
            # 只在主进程更新进度条和wandb
            if self.accelerator.is_main_process:
                if hasattr(dataloader_iterator, "set_postfix"):
                    dataloader_iterator.set_postfix({"loss": f"{current_loss:.4f}"})
                
                # 减少wandb调用频率
                if batch_idx % log_interval == 0 or batch_idx == len(self.dataloader) - 1:
                    wandb.log({
                        "train_loss": current_loss,
                        "step": self.global_step,
                        "epoch_progress": epoch + (batch_idx / len(self.dataloader)),
                        "gpu_memory_GB": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                        "batch": batch_idx
                    })

        return total_loss / len(self.dataloader)

    def run(self):
        # 记录总训练时间
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            avg_loss = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start
            
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
                
                # 记录每个epoch的摘要指标
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch_loss": avg_loss,
                    "epoch_time_seconds": epoch_time,
                    "best_loss": min(avg_loss, self.best_loss),
                })
                
                self._save_checkpoint(epoch, avg_loss)
                self.best_loss = min(avg_loss, self.best_loss)
        
        # 结束后保存最终模型（仅主进程）
        if self.accelerator.is_main_process:
            torch.save(self.model.module.state_dict(), "final_model.pth")
            
            # 计算总训练时间
            total_time = time.time() - start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            
            # 最终摘要
            wandb.log({
                "final_loss": self.best_loss,
                "training_time_hours": hours,
                "training_time_minutes": minutes,
                "total_epochs_completed": self.epochs - self.start_epoch
            })
            
            # 保存最终模型到wandb
            wandb.save("final_model.pth")
            print(f"Training complete in {hours}h {minutes}m. Best loss: {self.best_loss:.4f}")

if __name__ == "__main__":
    # 使大规模训练时的显存分配更灵活
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 解析命令行参数
    args = parse_args()
    
    # 创建并运行训练系统
    system = TrainingSystem(args)
    system.run()
