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

# Import the GPT2 model for distillation
from gpt2 import GPT2PixelAR

# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file accelerate_config.yaml distil_lora.py

# 允许加载部分损坏/截断的图像（如果可能）
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 设置日志目录
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"distil_lora_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")

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
    
    def __init__(self, root_dir, patch_size=16, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.patch_size = patch_size
        self._init_paths()
        
        # Default transform if none provided - 移除 Resize 操作
        if transform is None:
            self.transform = T.Compose([
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
    
    def __getitem__(self, idx):
        """Get image and prepare for training"""
        path = self.paths[idx]
        try:
            # 加载并转换图像 - 不改变尺寸
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img)  # 直接转换，不调整大小
            
            # 随机提取patch
            patch = self.random_patchify(img_tensor)
            
            # 将pixel转换为整数，并准备为序列
            pixel_values = patch.long().clamp(0, 255)
            c, h, w = pixel_values.shape
            seq_len = c * h * w
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
        
    def random_patchify(self, img_tensor):
        """从图像中随机提取patch，处理不同尺寸图像"""
        import random
        c, h, w = img_tensor.shape
        
        # 检查图像是否足够大能提取patch
        if h < self.patch_size or w < self.patch_size:
            # 如果图像太小，填充到最小尺寸
            logger.warning(f"图像尺寸太小 ({h}x{w})，添加填充至最小尺寸")
            padder = nn.ZeroPad2d((0, max(0, self.patch_size - w), 0, max(0, self.patch_size - h)))
            img_tensor = padder(img_tensor)
            # 更新尺寸
            c, h, w = img_tensor.shape
        
        # 随机选择起始位置
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        
        # 提取patch
        patch = img_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
        return patch

class GPT2LoraFineTuner:
    """使用LoRA对GPT2蒸馏模型进行微调的类"""
    
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
                name=args.wandb_run_name or f"distil-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(args)
            )
            self.accelerator.init_trackers("llama-compression")
            # 在wandb中记录模型架构信息
            wandb.run.notes = f"使用LoRA微调蒸馏后的GPT2模型，秩={args.lora_rank}"
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
        (
            self.model,
            self.optimizer, 
            self.lr_scheduler,
            self.train_dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer, 
            self.lr_scheduler,
            self.train_dataloader
        )

        self.base_model.prob = self.base_model.prob.to(self.device)
        if hasattr(self.accelerator, "mixed_precision") and self.accelerator.mixed_precision == "bf16":
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
        """初始化GPT2模型与LoRA配置"""
        logger.info("正在初始化蒸馏后的GPT2模型...")
        
        # 加载GPT2模型
        self.base_model = GPT2PixelAR(model_path=self.args.gpt2_model_dir)
        
        # 加载蒸馏检查点
        checkpoint = torch.load(self.args.model_checkpoint, map_location="cpu")
        
        # 处理可能的state_dict格式
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
            
        # 处理可能的module前缀
        if all(k.startswith("module.") for k in checkpoint.keys()):
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}
                
        # 加载状态字典
        self.base_model.load_state_dict(checkpoint, strict=False)
        self.base_model.eval()  # 设置为评估模式
        logger.info("蒸馏GPT2模型加载成功")
        
        # 冻结prob层参数
        self.base_model.prob.requires_grad_(False)
        logger.info("已冻结prob层参数")
        
        # 修正: 提取GPT2模型组件 - 使用正确的属性名
        self.gpt2_model = self.base_model.gpt2
        
        # 配置LoRA - 调整为GPT2架构
        logger.info("应用LoRA配置到GPT2...")
        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=[
                # 注意力机制部分
                "attn.c_attn",         # 查询/键/值投影 (最关键)
                "attn.c_proj",         # 注意力输出投影
                
                # MLP部分
                "mlp.c_fc",            # MLP中间层投影
                "mlp.c_proj",          # MLP输出投影
                
                # 其他层 (可选添加)
                "ln_1.weight",         # 第一层归一化权重
                "ln_2.weight",         # 第二层归一化权重
                "ln_f.weight",         # 最终层归一化权重
                "wte.weight",          # 词嵌入表
                "wpe.weight",          # 位置嵌入表
            ]
        )
        
        # 创建LoRA模型
        self.model = get_peft_model(self.gpt2_model, lora_config)
        trainable_params = self.model.print_trainable_parameters()
        logger.info(f"创建的GPT2 LoRA模型: {trainable_params}")
        
        # 启用梯度检查点以节省内存（如果请求）
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # 添加详细的参数统计
        total_params = 0
        trainable_params = 0
        lora_params_dict = {}

        # 遍历所有模块和参数
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            
            # 统计总参数量
            total_params += param_count
            
            # 统计可训练参数量
            if param.requires_grad:
                trainable_params += param_count
                
                # 提取LoRA适配器名称（从参数名中获取）
                module_name = name.split('.')[0] if '.' in name else name
                if 'lora' in name:
                    adapter_name = name.split('lora_')[1].split('.')[0] if 'lora_' in name else 'unknown'
                    full_name = f"{module_name}.lora_{adapter_name}"
                    
                    # 累计每个适配器的参数量
                    if full_name not in lora_params_dict:
                        lora_params_dict[full_name] = 0
                    lora_params_dict[full_name] += param_count

        # 打印详细统计信息
        logger.info(f"模型总参数量: {total_params:,}")
        logger.info(f"可训练参数量: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info("LoRA适配器参数统计:")
        for adapter_name, param_count in sorted(lora_params_dict.items()):
            logger.info(f"  - {adapter_name}: {param_count:,} 参数")

        # 记录到wandb
        if self.args.use_wandb and self.accelerator.is_main_process:
            wandb.config.update({
                "total_params": total_params,
                "trainable_params": trainable_params,
                "trainable_percent": 100 * trainable_params / total_params
            })
    
    def _init_dataset(self):
        """初始化数据集和数据加载器"""
        logger.info("加载数据集...")
        
        # 创建训练数据集 - 移除size参数
        self.train_dataset = RobustImageDataset(
            root_dir=self.args.train_dataset,
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
            artifact = wandb.Artifact(f"gpt2_lora_model", type="model")
            artifact.add_dir(os.path.join(self.args.output_dir, "best_model"))
            wandb.log_artifact(artifact)
            # 结束wandb跟踪器
            self.accelerator.end_training()
        
        return self.best_loss
    
    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.base_model.prob.eval()  # 确保prob层为评估模式，不参与训练
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
                # 使用LoRA模型获取隐藏状态
                outputs = self.model(input_ids=batch["input_ids"])
                
                # 从输出中获取隐藏状态
                hidden_states = outputs.last_hidden_state
                
                logits = self.base_model.prob(hidden_states)
                
                # 计算损失
                loss = F.cross_entropy(
                    logits.reshape(-1, 256),  # [batch*seq_len, num_classes]
                    batch["labels"].reshape(-1)  # [batch*seq_len]
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
            
            # 更新全局步骤
            self.global_step += 1
            
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
        
        # 保存prob层权重
        prob_weights_path = os.path.join(checkpoint_dir, "prob_layer.pt")
        torch.save(self.base_model.prob.state_dict(), prob_weights_path)
        
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
            artifact = wandb.Artifact(f"gpt2_lora_model_{self.global_step}", type="model")
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
        
        # 加载prob层权重
        prob_weights_path = os.path.join(checkpoint_dir, "prob_layer.pt")
        if os.path.exists(prob_weights_path):
            self.base_model.prob.load_state_dict(torch.load(prob_weights_path))
            logger.info("Output layer weights loaded successfully")
        
        # 更新最后检查点保存时间
        self.last_checkpoint_time = time.time()
        
        logger.info(f"从epoch {self.start_epoch}加载的检查点，loss {self.best_loss:.4f}")

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用LoRA对蒸馏的GPT2图像压缩模型进行微调")
    
    # 模型参数
    parser.add_argument("--model_checkpoint", type=str, default="/remote-home/wufeiyang/3_distill/4_distill_140_epoch/best_ce_model/model.pth",
                        help="蒸馏GPT2模型检查点路径")
    parser.add_argument("--gpt2_model_dir", type=str, default="/remote-home/wufeiyang/gpt2_model",
                        help="GPT2模型配置目录路径")
    
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
    parser.add_argument("--batch_size", type=int, default=24,
                        help="每个GPU的训练批量大小")
    parser.add_argument("--num_epochs", type=int, default=100,
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
    
    # Accelerator参数
    parser.add_argument("--use_deepspeed", action="store_true", default=True,
                        help="使用DeepSpeed插件")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子以确保可重现性")
    parser.add_argument("--output_dir", type=str, default="/remote-home/wufeiyang/gpt2_lora_output",
                        help="保存输出文件的目录")
    
    return parser.parse_args()

def main():
    """主函数"""
    # 使大规模训练时的显存分配更灵活
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 解析参数
    args = get_args()
    
    # 创建并运行微调器
    fine_tuner = GPT2LoraFineTuner(args)
    best_loss = fine_tuner.train()
    
    logger.info(f"训练完成。最终最佳损失: {best_loss:.4f}")
    
    return best_loss

if __name__ == "__main__":
    main()
