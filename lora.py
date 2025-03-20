import os
import glob
import logging
import argparse
from datetime import datetime
import random
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

# 使用 Accelerate 和 DeepSpeed
from accelerate import Accelerator, DeepSpeedPlugin

# PEFT for LoRA
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

# Import your original model
from main import SafeLlamaPixelAR

# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file accelerate_config.yaml lora.py

# 允许加载部分损坏/截断的图像（如果可能）
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 设置日志目录
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")

# Set up logging with both file and console output
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustImageDataset(Dataset):
    """Dataset for image compression fine-tuning with robust handling"""
    
    def __init__(self, root_dir, size=256, patch_size=16, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.size = size
        self.patch_size = patch_size
        self._init_paths()
        
        # Default transform if none provided
        if transform is None:
            self.transform = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Lambda(lambda x: x * 255)  # Scale to [0, 255]
            ])
        else:
            self.transform = transform
            
    def _init_paths(self):
        """Initialize image paths"""
        self.paths = sorted(glob.glob(os.path.join(self.root_dir, "*.jpg")))
        logger.info(f"找到 {len(self.paths)} 个图像文件")

    def __len__(self):
        return len(self.paths)
        
    def random_patchify(self, img_tensor):
        """随机从图像中提取patch"""
        import random
        c, h, w = img_tensor.shape
        
        # 随机选择起始位置
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        
        # 提取patch
        patch = img_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
        return patch
        
    def __getitem__(self, idx):
        """Get image and prepare for training"""
        path = self.paths[idx]
        try:
            # Load and transform image
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img)
            
            # 随机提取patch
            patch = self.random_patchify(img_tensor)
            
            # 将pixel转换为整数，并准备为序列
            pixel_values = patch.long().clamp(0, 255)
            b, h, w = pixel_values.shape
            seq_len = b * h * w
            pixel_ids = pixel_values.reshape(seq_len)
            
            # 为自回归预测准备输入和标签
            input_ids = pixel_ids[:-1]  # 除最后一个像素外的所有像素
            labels = pixel_ids[1:]      # 除第一个像素外的所有像素
            
            return {
                "input_ids": input_ids, 
                "labels": labels,
                "pixel_values": pixel_values,
            }
        except Exception as e:
            logger.warning(f"加载图像 {path} 出错: {str(e)}，跳过...")
            # 出错时使用下一个图像代替
            return self[(idx + 1) % len(self)]

class LoraFineTuner:
    """使用Accelerate和DeepSpeed进行LoRA微调的类"""
    
    def __init__(self, args):
        """
        初始化微调器
        
        Args:
            args: 命令行参数
        """
        self.args = args
        # 创建加速器
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            log_with="wandb" if args.use_wandb else None,
            deepspeed_plugin=DeepSpeedPlugin(
                zero_stage=2,
                offload_optimizer_device="none"
            ) if args.use_deepspeed else None
        )
        # 初始化wandb
        if args.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                entity="feiyangwu0309-sjtu-icec",
                project="llama-compression", 
                name=args.wandb_run_name or f"lora-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(args)
            )
            self.accelerator.init_trackers("llama-compression")
            # 在wandb中记录模型架构信息
            wandb.run.notes = f"使用LoRA微调LLaMA压缩模型，秩={args.lora_rank}"
            logger.info(f"已初始化wandb，项目：llama-compression")
            
        
        # 设置设备
        self.device = self.accelerator.device
        logger.info(f"使用设备: {self.device}")
        
        # 设置随机种子以确保可重现性
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            
        # 创建输出目录
        if self.accelerator.is_main_process and not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        # 初始化模型、数据集、优化器
        self._init_model()
        self._init_dataset()
        self._init_optimizer()
        
        # 将模型、优化器和数据加载器包装到加速器中
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader
        )
        
        # 确保基础模型在与LoRA模型相同的设备上且使用相同数据类型
        self.base_model = self.base_model.to(self.accelerator.device)
        self.base_model = self.base_model.to(dtype=torch.bfloat16)
        
        # 特别确保prob层也转换为bfloat16以避免类型不匹配
        self.base_model.prob = self.base_model.prob.to(self.accelerator.device)
        self.base_model.prob = self.base_model.prob.to(dtype=torch.bfloat16)
        
        # 训练状态
        self.global_step = 0
        self.best_loss = float("inf")
        self.start_epoch = 0
        self.last_checkpoint_time = time.time()  # 跟踪上次保存检查点的时间
        
        # 如果存在检查点，加载它
        if args.resume_from_checkpoint:
            self._load_checkpoint(args.resume_from_checkpoint)
        
    def _init_model(self):
        """初始化模型与LoRA配置"""
        logger.info("正在初始化模型...")
        
        # 加载基础模型
        self.base_model = SafeLlamaPixelAR(model_path=self.args.model_dir)
        self.base_model = self.base_model.to(self.accelerator.device)
        self.base_model = self.base_model.to(dtype=torch.bfloat16)

        # 加载检查点
        checkpoint = torch.load(self.args.model_checkpoint, map_location="cpu")
        
        # 处理可能的state_dict格式
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
            
        # 处理可能的module前缀
        if all(k.startswith("module.") for k in checkpoint.keys()):
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}
            
        # 重命名键以匹配模型结构
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("llama.model."):
                new_key = key.replace("llama.model.", "llama.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
                
        # 加载状态字典
        self.base_model.load_state_dict(new_state_dict, strict=False)
        logger.info("基础模型加载成功")
        
        # 提取LLaMA部分进行微调
        self.llama_model = self.base_model.llama
        
        # 配置LoRA
        logger.info("应用LoRA配置...")
        lora_config = LoraConfig(
            r=self.args.lora_rank,               # LoRA注意力维度
            lora_alpha=self.args.lora_alpha,     # LoRA缩放因子
            lora_dropout=self.args.lora_dropout, # Dropout概率
            # bias="none",                         # 不使用偏置参数
            # task_type="CAUSAL_LM",               # 自回归语言模型任务类型
            target_modules=[  
                # 现有层
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "down_proj", "up_proj",
                # 添加层归一化和嵌入层
                "input_layernorm.weight", "post_attention_layernorm.weight",
                "norm.weight",  # 最终归一化层
                "embed_tokens.weight"  # 词嵌入
            ]
        )
        
        # 创建LoRA模型
        self.model = get_peft_model(self.llama_model, lora_config)
        trainable_params = self.model.print_trainable_parameters()
        logger.info(f"创建的LoRA模型: {trainable_params}")
        
        # 启用梯度检查点以节省内存（如果请求）
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def _init_dataset(self):
        """初始化数据集和数据加载器"""
        logger.info("加载数据集...")
        
        # 创建训练数据集
        self.train_dataset = RobustImageDataset(
            root_dir=self.args.train_dataset,
            size=256,
            patch_size=16
        )
        
        logger.info(f"训练数据集: {len(self.train_dataset)} 图像")
        
        # 创建数据加载器
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        # 筛选可训练参数（仅LoRA）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 计算训练步数
        train_steps = len(self.train_dataloader) * self.args.num_epochs
        
        # 创建带预热的学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.args.learning_rate,
            total_steps=train_steps,
            pct_start=self.args.warmup_ratio,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        logger.info(f"已初始化优化器和调度器，训练步数: {train_steps}")
        
    def train(self):
        """运行完整的训练循环"""
        logger.info("开始训练循环...")
        
        # 跟踪指标以进行绘图
        train_losses = []
        
        # 训练循环
        for epoch in range(self.start_epoch, self.args.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            # 训练一个epoch
            train_loss = self._train_epoch(epoch)
            train_losses.append(train_loss)
            
            # 记录指标
            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
                       
            # 记录到wandb
            if self.args.use_wandb and self.accelerator.is_main_process:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch_loss": train_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # 如果是最佳损失，保存检查点
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self._save_checkpoint("best_model", epoch)
                logger.info(f"已保存新的最佳模型，损失值: {train_loss:.4f}")
            
            # 总是保存最新的检查点
            self._save_checkpoint(f"checkpoint_epoch_{epoch+1}", epoch)
            
        logger.info(f"训练完成。最佳损失: {self.best_loss:.4f}")
        
        # 最终保存
        self._save_checkpoint("final_model", self.args.num_epochs-1)
        
        # 结束wandb运行
        if self.args.use_wandb and self.accelerator.is_main_process:
            # 记录最终结果
            wandb.log({"best_loss": self.best_loss})
            # 上传最终模型文件
            artifact = wandb.Artifact(f"lora_model", type="model")
            artifact.add_dir(os.path.join(self.args.output_dir, "best_model"))
            wandb.log_artifact(artifact)
            # 结束wandb跟踪器
            self.accelerator.end_training()
        
        return self.best_loss
    
    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        checkpoint_interval = self.args.checkpoint_interval * 60  # 分钟转秒
        
        # 使用tqdm进行进度条显示（只在主进程上）
        if self.accelerator.is_main_process:
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
        else:
            pbar = self.train_dataloader
        
        for step, batch in enumerate(pbar):
            # 确保batch中的数据在正确的设备上
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            with self.accelerator.autocast():
                outputs = self.model(
                    input_ids=batch["input_ids"]
                )
                
                # 从输出中获取隐藏状态
                hidden_states = outputs.last_hidden_state
                
                # 使用原始模型的prob层计算logits
                logits = self.base_model.prob(hidden_states)
                
                # 使用原始模型的损失函数计算损失
                loss = self.base_model.loss_fn(
                    logits.view(-1, 256),
                    batch["labels"].view(-1)
                )
            
            # 反向传播
            self.accelerator.backward(loss)
            
            # 裁剪梯度
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), 
                self.args.max_grad_norm
            )
            
            # 更新权重
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            # 更新进度条
            loss_item = loss.detach().float()
            # 确保所有进程获得相同的损失值
            loss_item = self.accelerator.gather(loss_item).mean().item()
            total_loss += loss_item
            avg_loss = total_loss / (step + 1)
            
            # 只在主进程上更新进度条
            if self.accelerator.is_main_process:
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # 记录每个步骤的指标到wandb
                if self.args.use_wandb and step % self.args.wandb_log_steps == 0:
                    wandb.log({
                        "step": self.global_step,
                        "step_loss": loss_item,
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch_progress": epoch + (step / len(self.train_dataloader)),
                        "avg_loss": avg_loss,
                        "gpu_memory": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    })
                    
                    # 每500步记录一次梯度分布
                    if self.global_step % 500 == 0:
                        trainable_params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                        if trainable_params:
                            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in trainable_params]))
                            wandb.log({"gradient_norm": grad_norm}, step=self.global_step)
            
            # 更新全局步骤
            self.global_step += 1
            
            # 根据时间保存检查点
            current_time = time.time()
            if current_time - self.last_checkpoint_time > checkpoint_interval:
                self._save_checkpoint(f"checkpoint_time_{int(current_time)}", epoch)
                logger.info(f"已保存时间检查点 (step {self.global_step})")
                self.last_checkpoint_time = current_time
            
            # 根据步数保存检查点
            if self.global_step % self.args.save_steps == 0:
                self._save_checkpoint(f"checkpoint_step_{self.global_step}", epoch)
                logger.info(f"已保存步骤检查点 (step {self.global_step})")
            
        # 确保所有进程都完成且损失值一致
        avg_loss = total_loss / len(self.train_dataloader)
        # 确保张量在正确的设备上
        return self.accelerator.gather(torch.tensor(avg_loss, device=self.accelerator.device)).mean().item()
    
    def _save_checkpoint(self, checkpoint_name, epoch):
        """保存模型检查点"""
        # 只在主进程上保存检查点
        if not self.accelerator.is_main_process:
            return
            
        # 创建检查点目录
        checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # 保存LoRA模型权重
        # 先展开模型
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save
        )
        
        # 保存训练状态
        training_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "args": vars(self.args),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        torch.save(training_state, state_path)
        
        logger.info(f"检查点已保存至 {checkpoint_dir}")
        
        # 将模型保存为wandb artifact
        if self.args.use_wandb and checkpoint_name == "best_model":
            artifact = wandb.Artifact(f"lora_model_{self.global_step}", type="model")
            artifact.add_dir(checkpoint_dir)
            wandb.log_artifact(artifact)
    
    def _load_checkpoint(self, checkpoint_dir):
        """加载模型检查点"""
        logger.info(f"正在从 {checkpoint_dir} 加载检查点...")
        
        # 加载训练状态
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(state_path):
            training_state = torch.load(state_path, map_location="cpu")
            self.start_epoch = training_state["epoch"] + 1
            self.global_step = training_state["global_step"]
            self.best_loss = training_state["best_loss"]
            
            # 加载优化器和学习率调度器状态
            if "optimizer_state" in training_state and hasattr(self, "optimizer"):
                try:
                    self.optimizer.load_state_dict(training_state["optimizer_state"])
                    logger.info("优化器状态已恢复")
                except Exception as e:
                    logger.warning(f"无法加载优化器状态: {e}")
                    
            if "scheduler_state" in training_state and hasattr(self, "lr_scheduler") and training_state["scheduler_state"]:
                try:
                    self.lr_scheduler.load_state_dict(training_state["scheduler_state"])
                    logger.info("学习率调度器状态已恢复")
                except Exception as e:
                    logger.warning(f"无法加载学习率调度器状态: {e}")
        
        # 加载LoRA权重
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model = PeftModel.from_pretrained(
            unwrapped_model,
            checkpoint_dir,
            is_trainable=True
        )
        
        # 更新最后检查点保存时间
        self.last_checkpoint_time = time.time()
        
        logger.info(f"从epoch {self.start_epoch}加载的检查点，loss {self.best_loss:.4f}")

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用LoRA为基于LLaMA的图像压缩模型进行微调")
    
    # 模型参数
    parser.add_argument("--model_checkpoint", type=str, default="/remote-home/wufeiyang/model_epoch_40.pth",
                        help="模型检查点路径")
    parser.add_argument("--model_dir", type=str, default="/remote-home/wufeiyang/saved_model",
                        help="模型配置目录路径")
    
    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA适应矩阵的秩")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA的缩放因子")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA层的Dropout概率")
    
    # 数据集参数
    parser.add_argument("--train_dataset", type=str, default="/remote-home/wufeiyang/dataset/flickr30k/Images",
                        help="训练数据集路径")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="图像patch大小")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8,
                        help="每个GPU的训练批量大小")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="训练轮次数")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="峰值学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW的权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="学习率预热的步骤比例")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪的最大范数")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="启用梯度检查点以节省内存")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="数据加载的工作进程数")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="从检查点恢复训练")
    
    # 检查点保存参数
    parser.add_argument("--checkpoint_interval", type=int, default=60,
                        help="每隔多少分钟保存一次时间检查点")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="每多少步保存一次步骤检查点")
    
    # Wandb参数
    parser.add_argument("--use_wandb", action="store_true", default=True,
                        help="是否使用wandb进行训练监控")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb运行名称")
    parser.add_argument("--wandb_log_steps", type=int, default=10,
                        help="每多少步记录一次wandb指标")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子以确保可重现性")
    parser.add_argument("--output_dir", type=str, default="./lora_output",
                        help="保存输出文件的目录")
    
    # Accelerator参数
    parser.add_argument("--use_deepspeed", action="store_true", default=True,
                        help="使用DeepSpeed插件")
    
    return parser.parse_args()

def main():
    """主函数"""
    # 使大规模训练时的显存分配更灵活
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 解析参数
    args = get_args()
    
    # 创建并运行微调器
    fine_tuner = LoraFineTuner(args)
    best_loss = fine_tuner.train()
    
    logger.info(f"训练完成。最终最佳损失: {best_loss:.4f}")
    
    return best_loss

if __name__ == "__main__":
    main()
