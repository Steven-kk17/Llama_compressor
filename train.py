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
import random
import numpy as np

# Accelerate & DeepSpeed
from accelerate import Accelerator, DeepSpeedPlugin

from main import SafeLlamaPixelAR

# Allow loading partially corrupted / truncated images if possible
ImageFile.LOAD_TRUNCATED_IMAGES = True

def custom_collate_fn(batch):
    """自定义collate_fn处理不同大小的图像"""
    patches = torch.cat([item[0] for item in batch], dim=0)  # patches已经是固定大小且相同形状
    # 图像保持原样，不进行批处理
    images = [item[1] for item in batch]
    return patches, images

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="训练LLaMA图像压缩模型")
    parser.add_argument("--epochs", type=int, default=180,
                        help="训练的总epoch数")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="每个GPU的批处理大小")
    parser.add_argument("--lr", type=float, default=1e-5,  # 更低的初始学习率
                        help="初始学习率")
    parser.add_argument("--seed", type=int, default=42,
                        help="训练的随机种子")
    parser.add_argument("--dataset", type=str, 
                        default="/remote-home/wufeiyang/dataset/flickr30k/Images",
                        help="训练数据集路径")
    parser.add_argument("--model_path", type=str,
                        default="/remote-home/wufeiyang/Llama_1B",
                        help="基础模型路径")
    parser.add_argument("--pretrained", type=str,
                        default="ckpt.pth",
                        help="预训练模型路径，直接加载权重")
    parser.add_argument("--wandb_name", type=str,
                        default=None,
                        help="wandb运行名称，不指定则自动生成")
    parser.add_argument("--start_epoch", type=int, default=80,
                        help="开始训练的epoch数")
    parser.add_argument("--target_lr", type=float, default=5e-5,
                        help="目标学习率")
    return parser.parse_args()

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置确定性选项
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")

# ---------- 增强数据集类 ----------
# 修改 RobustImageDataset 类的 transform

class RobustImageDataset(Dataset):
    def __init__(self, root_dir, patch_size=16):
        super().__init__()
        self.root_dir = root_dir
        # 移除 size 参数，不再调整图像大小
        self.patch_size = patch_size
        self._init_paths()
        
        # 修改 transform，移除 Resize 操作
        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x * 255)  # 仅将像素值映射到 [0, 255]
        ])

    def _init_paths(self):
        self.paths = sorted(glob.glob(os.path.join(self.root_dir, "*.jpg")))
        print(f"Found {len(self.paths)} image files")

    def __len__(self):
        return len(self.paths)

    def random_patchify(self, img_tensor, num_patches=1):
        c, h, w = img_tensor.shape
        patches = []
        
        # 确保图像足够大能提取 patch
        if h < self.patch_size or w < self.patch_size:
            # 如果图像太小，先进行临时调整
            print(f"Warning: Image too small ({h}x{w}), padding to minimum size")
            padder = nn.ZeroPad2d((0, max(0, self.patch_size - w), 0, max(0, self.patch_size - h)))
            img_tensor = padder(img_tensor)
            # 更新尺寸
            c, h, w = img_tensor.shape
        
        for _ in range(num_patches):
            # 从原始尺寸图像随机提取 patch
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            patch = img_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
            patches.append(patch)
        return torch.stack(patches, dim=0)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img)  # 直接转换，不调整大小
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
        self.lr = args.lr  # 使用命令行参数中的学习率
        self.target_lr = args.target_lr  # 目标学习率
        self.patch_size = 16  # 保留 patch 大小
        self.batch_size_per_gpu = args.batch_size
        self.checkpoint_path = "training_checkpoint.pth"
        self.best_model_path = "best_model.pth"  # 新增: 保存最佳模型的路径
        self.model_path = args.model_path
        self.pretrained_path = args.pretrained
        
        # 设置起始epoch和预热期
        self.start_epoch = args.start_epoch
        self.warmup_epochs = 5  # 从当前起始epoch开始预热5个epoch
        
        # 全局步数，用于wandb记录 - 根据起始epoch估算
        self.global_step = self.start_epoch * 1171  # 假设每个epoch约1171个batch
        
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
                    "initial_learning_rate": self.lr,
                    "target_learning_rate": self.target_lr,
                    "model": "SafeLlamaPixelAR",
                    "patch_size": self.patch_size,
                    "high_res": True,  # 添加标记表示使用高分辨率
                    "pretrained_model": args.pretrained,
                    "start_epoch": self.start_epoch
                }
            )
            print(f"Wandb initialized with run name: {run_name}")
        
        # 初始化组件
        self._init_dataset()
        self._init_model()
        self._init_optimizer()
        
        # 从预训练模型直接加载权重
        self._load_pretrained_model()

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

        # 初始化最佳loss为一个较高的值
        self.best_loss = 4.2937  # 基于之前训练的最佳值
    
    def _load_pretrained_model(self):
        """直接从预训练模型文件加载权重，不需要处理checkpoint格式"""
        if not os.path.exists(self.pretrained_path):
            if self.accelerator.is_main_process:
                print(f"预训练模型 {self.pretrained_path} 不存在，将从头开始训练")
            return
            
        if self.accelerator.is_main_process:
            print(f"从预训练模型 {self.pretrained_path} 加载权重")
        
        # 直接加载模型权重
        try:
            # 加载权重
            pretrained_weights = torch.load(self.pretrained_path, map_location="cpu")
            
            # 检查是否是状态字典还是包含状态字典的字典
            if isinstance(pretrained_weights, dict) and "state_dict" in pretrained_weights:
                pretrained_weights = pretrained_weights["state_dict"]
                
            # 检查是否有"module."前缀并处理
            if all(k.startswith("module.") for k in pretrained_weights.keys()):
                pretrained_weights = {k[7:]: v for k, v in pretrained_weights.items()}
                
            # 加载到模型
            self.model.load_state_dict(pretrained_weights, strict=False)
            
            if self.accelerator.is_main_process:
                print("预训练模型加载成功")
                print(f"将从epoch {self.start_epoch} 开始训练到 epoch {self.epochs}")
                print(f"初始学习率设置为 {self.lr:.6f}")
                
        except Exception as e:
            if self.accelerator.is_main_process:
                print(f"加载预训练模型时出错: {e}")
                print("将从头开始训练")
    
    def _init_dataset(self):
        dataset = RobustImageDataset(
            self.root_dir,
            patch_size=self.patch_size
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size_per_gpu,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            collate_fn=custom_collate_fn  # 添加自定义collate函数
        )

    def _init_model(self):
        # 建立基础模型
        base_model = SafeLlamaPixelAR(
            model_path=self.model_path
        )
        self.model = base_model


    def _init_optimizer(self):
        # 使用较低的初始学习率
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,  # 使用较低的初始学习率
            weight_decay=0.01,
            betas=(0.9, 0.95)  # 调整beta值以增强动量
        )

    def _save_checkpoint(self, epoch, avg_loss):
        if self.accelerator.is_main_process:
            checkpoint = {
                "epoch": epoch,
                "model_state": self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_loss": min(avg_loss, self.best_loss),
                "global_step": self.global_step,
                "last_lr": self.optimizer.param_groups[0]['lr'],  # 保存当前学习率
            }
            torch.save(checkpoint, self.checkpoint_path)
            
            # 如果当前loss是最好的，保存最佳模型
            if avg_loss < self.best_loss:
                model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                torch.save(model_state, self.best_model_path)
                print(f"New best loss: {avg_loss:.4f}, saved best model")
                if wandb.run is not None:
                    wandb.log({"new_best_model": True, "best_loss_updated": avg_loss})
            
            # 每10个epoch保存一个模型版本
            if (epoch + 1) % 10 == 0:
                epoch_path = f"model_epoch_{epoch+1}.pth"
                model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                torch.save(model_state, epoch_path)
                print(f"Saved model at epoch {epoch+1} to {epoch_path}")
                if wandb.run is not None:
                    wandb.save(epoch_path)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        log_interval = 50  # 每50个batch记录一次wandb（降低API调用频率）
        
        # 计算从当前epoch开始的预热进度
        relative_epoch = epoch - self.start_epoch
        
        # 在预热期间动态调整学习率
        if relative_epoch < self.warmup_epochs:
            progress = (relative_epoch + 1) / self.warmup_epochs
            current_lr = self.lr + progress * (self.target_lr - self.lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            if self.accelerator.is_main_process:
                print(f"预热阶段 {relative_epoch+1}/{self.warmup_epochs}, 学习率: {current_lr:.6f}")
        elif relative_epoch == self.warmup_epochs:
            # 预热结束后设置为目标学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.target_lr
            if self.accelerator.is_main_process:
                print(f"预热完成，设置目标学习率: {self.target_lr}")

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
            
            # patches 形状已经是 [B*n, c, ph, pw]，不需要再调整形状
            input_patches = patches  # 直接使用，不需要 view 操作
            
            with self.accelerator.autocast():
                _, ce_loss = self.model(input_patches, None)
                ce_loss = ce_loss.mean()

            self.optimizer.zero_grad()
            self.accelerator.backward(ce_loss)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 记录当前固定学习率
            if self.accelerator.is_main_process and batch_idx % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                wandb.log({"learning_rate": current_lr}, step=self.global_step)

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
            model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
            torch.save(model_state, "final_model.pth")
            
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
            wandb.save("best_model.pth")  # 也保存最佳模型到wandb
            print(f"Training complete in {hours}h {minutes}m. Best loss: {self.best_loss:.4f}")

if __name__ == "__main__":
    # 使大规模训练时的显存分配更灵活
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建并运行训练系统
    system = TrainingSystem(args)
    system.run()
