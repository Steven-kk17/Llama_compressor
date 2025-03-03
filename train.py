# filepath: /home/steven/code/llama_compression/train.py
import os
import glob
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from main import SafeLlamaPixelAR
from PIL import ImageFile

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
        """动态加载路径并过滤损坏文件"""
        self.paths = []
        raw_paths = sorted(glob.glob(os.path.join(self.root_dir, "*.jpg")))
        
        # 预验证图像文件
        # for path in tqdm(raw_paths, desc="Validating images"):
        #     try:
        #         with Image.open(path) as img:
        #             img.verify()
        #         self.paths.append(path)
        #     except (IOError, OSError, Image.DecompressionBombError) as e:
        #         print(f"Removing corrupted file: {path} ({str(e)})")
        
        # Since validation is commented out, just use all paths
        self.paths = raw_paths
        print(f"Found {len(self.paths)} image files")

    def __len__(self):
        return len(self.paths)

    def patchify(self, img_tensor):
        c, h, w = img_tensor.shape
        patches = []
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = img_tensor[:, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        return torch.stack(patches, dim=0)
    
    def random_patchify(self, img_tensor, num_patches=1):
        """随机采样 16×16 patch，用于训练"""
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
        """读取图像并随机采 16×16 patch，不可读则跳过。"""
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img)
            # 使用随机采样函数
            patches = self.random_patchify(img_tensor, num_patches=16)  
            return patches, img_tensor
        except Exception as e:
            print(f"Error loading {path}: {str(e)}, skipping...")
            # 跳过损坏样本
            return self[(idx + 1) % len(self)]

# ---------- 训练系统 ----------
class TrainingSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = "training_checkpoint.pth"
        
        # 训练参数
        self.root_dir = "/remote-home/wufeiyang/dataset/flickr30k/Images"
        self.epochs = 10
        self.lr = 1e-4
        self.size = 256
        self.patch_size = 16
        self.batch_size_per_gpu = 1
        
        # 初始化组件
        self._init_dataset()
        self._init_model()
        self._init_optimizer()
        
    def _init_dataset(self):
        self.dataset = RobustImageDataset(
            self.root_dir, 
            size=self.size,
            patch_size=self.patch_size
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu * torch.cuda.device_count(),
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            drop_last=True
        )
    
    def _init_model(self):
        # 基础模型
        self.base_model = SafeLlamaPixelAR(
            model_path="/remote-home/wufeiyang/saved_model"
        )
        
        # 断点恢复
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            self.base_model.load_state_dict(checkpoint["model_state"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_loss = checkpoint["best_loss"]
            print(f"Loaded checkpoint from epoch {self.start_epoch}, loss {self.best_loss:.4f}")
        else:
            self.start_epoch = 0
            self.best_loss = float("inf")
        
        # 多GPU支持
        if torch.cuda.device_count() >= 1:
            self.model = nn.DataParallel(self.base_model.to(self.device))
        self.model = self.model.bfloat16()
    
    def _init_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=0.01
        )
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    def _save_checkpoint(self, epoch, avg_loss):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.base_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_loss": min(avg_loss, self.best_loss),
        }
        torch.save(checkpoint, self.checkpoint_path)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        # 进度条
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx, (patches, _) in enumerate(progress_bar):
            # 数据准备
            b, n, c, ph, pw = patches.shape
            input_patches = patches.view(b*n, c, ph, pw).to(self.device, dtype=torch.bfloat16)
            
            # 前向传播
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, ce_loss = self.model(input_patches, None)
                ce_loss = ce_loss.mean()
            
            # 反向传播
            ce_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 记录损失
            total_loss += ce_loss.item()
            progress_bar.set_postfix({
                "loss": f"{total_loss/(batch_idx+1):.4f}",
                "mem": f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB"
            })
        
        return total_loss / len(self.dataloader)
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
            self._save_checkpoint(epoch, avg_loss)
        
        # 最终保存
        torch.save(self.base_model.state_dict(), "final_model.pth")

if __name__ == "__main__":
    # 可以尝试让 PyTorch 在 CUDA 中动态分配内存
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    system = TrainingSystem()
    system.run()
