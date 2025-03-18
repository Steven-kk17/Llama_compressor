import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
import wandb
import math
import argparse
import logging
from datetime import datetime

# Accelerate & DeepSpeed
from accelerate import Accelerator, DeepSpeedPlugin

# Import your models
from main import SafeLlamaPixelAR
from gpt2 import GPT2PixelAR


# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config.yaml distillation.py



# Allow loading partially corrupted / truncated images if possible
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"distillation_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")

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
    """Dataset for knowledge distillation with robust handling"""
    
    def __init__(self, root_dir, size=256, patch_size=16):
        super().__init__()
        self.root_dir = root_dir
        self.size = size
        self.patch_size = patch_size
        self._init_paths()
        
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 255)  # Scale to [0, 255]
        ])
            
    def _init_paths(self):
        """Initialize image paths"""
        self.paths = sorted(glob.glob(os.path.join(self.root_dir, "*.jpg")))
        logger.info(f"Found {len(self.paths)} image files")

    def __len__(self):
        return len(self.paths)
        
    def random_patchify(self, img_tensor, num_patches=1):
        """Extract random patches from image"""
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
        """Get image and prepare for training"""
        path = self.paths[idx]
        try:
            # Load and transform image
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img)
            
            # Extract random patches
            patches = self.random_patchify(img_tensor, num_patches=1)
            return patches
        except Exception as e:
            logger.warning(f"Error loading image {path}: {str(e)}, skipping...")
            # On error, use next image
            return self[(idx + 1) % len(self)]


class DistillationTrainer:
    """Knowledge Distillation from LLaMA to GPT2 for image compression"""
    
    def __init__(self, args):
        """Initialize the distillation trainer"""
        if torch.cuda.is_available():
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}, " 
                        f"Memory: {torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB")

        self.args = args
        
        # Create accelerator
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            log_with="wandb" if args.use_wandb else None,
            deepspeed_plugin=DeepSpeedPlugin(
                zero_stage=2,
                offload_optimizer_device="none"
            ) if args.use_deepspeed else None
        )
        
        # Initialize wandb
        if args.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                entity="feiyangwu0309-sjtu-icec",
                project="llama-compression", 
                name=args.wandb_run_name or f"distillation-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(args)
            )
            self.accelerator.init_trackers("llama-compression")
            wandb.run.notes = f"Distilling LLaMA compression model to GPT2"
            logger.info(f"Initialized wandb, project: llama-compression")
        
        # Set device
        self.device = self.accelerator.device
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            
        # Create output directory
        if self.accelerator.is_main_process and not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        # Initialize models, dataset, optimizer
        self._init_models()
        self._init_dataset()
        self._init_optimizer()
        
        # Prepare with accelerator
        (
            self.student_model, 
            self.optimizer, 
            self.lr_scheduler,
            self.train_dataloader
        ) = self.accelerator.prepare(
            self.student_model, 
            self.optimizer, 
            self.lr_scheduler,
            self.train_dataloader
        )
        
        # Convert teacher model to same precision as student
        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model = self.teacher_model.to(dtype=torch.bfloat16)
        
        # Training state
        self.global_step = 0
        self.best_loss = float("inf")
        self.start_epoch = 0
        
        # Resume from checkpoint if specified
        if args.resume_from_checkpoint:
            self._load_checkpoint(args.resume_from_checkpoint)
        
    def _init_models(self):
        """Initialize teacher (LLaMA) and student (GPT2) models with LoRA enhanced teacher"""
        logger.info("Initializing models...")
        
        # 加载教师模型 (LLaMA)
        logger.info(f"Loading teacher model from {self.args.teacher_model_path}")
        self.teacher_model = SafeLlamaPixelAR(model_path=self.args.llama_model_dir)
        
        # 加载基础教师模型检查点
        teacher_checkpoint = torch.load(self.args.teacher_model_path, map_location="cpu")
        
        # 处理检查点格式
        if isinstance(teacher_checkpoint, dict) and "state_dict" in teacher_checkpoint:
            teacher_checkpoint = teacher_checkpoint["state_dict"]
            
        # 处理模块前缀
        if all(k.startswith("module.") for k in teacher_checkpoint.keys()):
            teacher_checkpoint = {k[7:]: v for k, v in teacher_checkpoint.items()}
            
        # 重命名键以匹配模型结构
        new_state_dict = {}
        for key, value in teacher_checkpoint.items():
            if key.startswith("llama.model."):
                new_key = key.replace("llama.model.", "llama.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
                
        # 加载教师模型权重
        self.teacher_model.load_state_dict(new_state_dict, strict=False)
        
        # 添加LoRA增强步骤
        if self.args.use_lora_teacher:
            logger.info(f"Enhancing teacher model with LoRA adapters from {self.args.lora_path}")
            
            try:
                from peft import PeftModel
                
                # 提取llama子模型
                llama_model = self.teacher_model.llama
                
                # 应用LoRA适配器到模型
                llama_model = PeftModel.from_pretrained(
                    llama_model,
                    self.args.lora_path,
                    is_trainable=False  # 只用于推理
                )
                
                # 合并LoRA权重到基础模型
                logger.info("Merging LoRA weights into teacher model...")
                llama_model = llama_model.merge_and_unload()
                logger.info("LoRA weights merged successfully")
                
                # 替换原始llama为LoRA增强版本
                self.teacher_model.llama = llama_model
            except Exception as e:
                logger.error(f"Failed to load or merge LoRA adapters: {e}")
                raise
        
        # 设置教师模型为评估模式
        self.teacher_model.eval()
        logger.info("Teacher model loaded successfully")
        
        # Initialize student model (GPT2)
        logger.info(f"Initializing student model from {self.args.gpt2_model_dir}")
        self.student_model = GPT2PixelAR(model_path=self.args.gpt2_model_dir)
        
        # Log model parameter counts
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        trainable_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        
        logger.info(f"Teacher model: {teacher_params:,} parameters")
        logger.info(f"Student model: {student_params:,} parameters")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/student_params:.2%} of student)")
        
    def _init_dataset(self):
        """Initialize dataset and dataloader"""
        logger.info("Loading dataset...")
        
        # Create training dataset
        self.train_dataset = RobustImageDataset(
            root_dir=self.args.train_dataset,
            size=256,
            patch_size=16
        )
        
        logger.info(f"Training dataset: {len(self.train_dataset)} images")
        
        # Create dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
    def _init_optimizer_with_custom_lrs(self, embed_lr_factor=1.5, top_lr_factor=0.8, bottom_lr_factor=0.4):
        """使用自定义学习率系数初始化优化器"""
        # 大部分代码与_init_optimizer相同，但使用传入的学习率系数
        
        # 记录可训练参数组
        param_groups = {}
        for name, param in self.student_model.named_parameters():
            if param.requires_grad:
                layer_name = name.split('.')[0] if '.' in name else name
                if layer_name not in param_groups:
                    param_groups[layer_name] = 0
                param_groups[layer_name] += param.numel()
        
        for group, count in param_groups.items():
            logger.info(f"Trainable params in {group}: {count:,}")
        
        # 使用分层学习率
        no_decay = ["bias", "LayerNorm.weight"]
        
        # 为每个参数确定唯一的组
        optimizer_grouped_params = []
        param_names_used = set()
        
        # 1. 嵌入和输出层
        for decay in [True, False]:
            params = []
            for n, p in self.student_model.named_parameters():
                if n in param_names_used:
                    continue
                    
                if any(nd in n for nd in ["wte", "prob"]):
                    if (not decay and any(nd in n for nd in no_decay)) or (decay and not any(nd in n for nd in no_decay)):
                        params.append(p)
                        param_names_used.add(n)
                        
            if params:
                optimizer_grouped_params.append({
                    "params": params,
                    "weight_decay": self.args.weight_decay if decay else 0.0,
                    "lr": self.args.learning_rate * embed_lr_factor  # 使用传入的因子
                })
        
        # 2. 顶部Transformer层
        for decay in [True, False]:
            params = []
            for n, p in self.student_model.named_parameters():
                if n in param_names_used:
                    continue
                    
                if any(f"h.{i}." in n for i in range(8, 12)):
                    if (not decay and any(nd in n for nd in no_decay)) or (decay and not any(nd in n for nd in no_decay)):
                        params.append(p)
                        param_names_used.add(n)
                        
            if params:
                optimizer_grouped_params.append({
                    "params": params,
                    "weight_decay": self.args.weight_decay if decay else 0.0,
                    "lr": self.args.learning_rate * top_lr_factor  # 使用传入的因子
                })
        
        # 3. 底部Transformer层
        for decay in [True, False]:
            params = []
            for n, p in self.student_model.named_parameters():
                if n in param_names_used:
                    continue
                    
                if any(f"h.{i}." in n for i in range(8)):
                    if (not decay and any(nd in n for nd in no_decay)) or (decay and not any(nd in n for nd in no_decay)):
                        params.append(p)
                        param_names_used.add(n)
                        
            if params:
                optimizer_grouped_params.append({
                    "params": params,
                    "weight_decay": self.args.weight_decay if decay else 0.0,
                    "lr": self.args.learning_rate * bottom_lr_factor  # 使用传入的因子
                })
        
        # 4. 检查未分组参数
        remaining_params = []
        for n, p in self.student_model.named_parameters():
            if n not in param_names_used and p.requires_grad:
                remaining_params.append(p)
                logger.warning(f"Parameter not in any group: {n}")
        
        if remaining_params:
            optimizer_grouped_params.append({
                "params": remaining_params,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate * 0.1
            })
        
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_params,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 计算训练步数并创建调度器
        total_steps = len(self.train_dataloader) * self.args.num_epochs
        
        # 对于阶段3，使用较少的预热比例
        if hasattr(self, 'global_step') and self.global_step > 0:
            warmup_ratio = 0.02  # 阶段3使用较短预热
        else:
            warmup_ratio = self.args.warmup_ratio
        
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[g["lr"] for g in optimizer_grouped_params],
            total_steps=total_steps,
            pct_start=warmup_ratio,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        logger.info(f"Initialized optimizer with {len(optimizer_grouped_params)} parameter groups")
        logger.info(f"Training steps: {total_steps}, warmup_ratio: {warmup_ratio}")


    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        # 记录可训练参数组以便清晰了解
        param_groups = {}
        for name, param in self.student_model.named_parameters():
            if param.requires_grad:
                layer_name = name.split('.')[0] if '.' in name else name
                if layer_name not in param_groups:
                    param_groups[layer_name] = 0
                param_groups[layer_name] += param.numel()
        
        for group, count in param_groups.items():
            logger.info(f"Trainable params in {group}: {count:,}")
        
        # 使用分层学习率
        no_decay = ["bias", "LayerNorm.weight"]
        
        # 为每个参数确定唯一的组
        optimizer_grouped_params = []
        param_names_used = set()
        
        # 1. 嵌入和输出层
        for decay in [True, False]:
            params = []
            for n, p in self.student_model.named_parameters():
                if n in param_names_used:
                    continue
                    
                if any(nd in n for nd in ["wte", "prob"]):
                    if (not decay and any(nd in n for nd in no_decay)) or (decay and not any(nd in n for nd in no_decay)):
                        params.append(p)
                        param_names_used.add(n)
                        
            if params:
                optimizer_grouped_params.append({
                    "params": params,
                    "weight_decay": self.args.weight_decay if decay else 0.0,
                    "lr": self.args.learning_rate * 1.5
                })
        
        # 2. 顶部Transformer层 (h.8 到 h.11) - 使用更精确的模式匹配
        for decay in [True, False]:
            params = []
            for n, p in self.student_model.named_parameters():
                if n in param_names_used:
                    continue
                    
                # 使用更精确的模式匹配，避免h.1与h.10等混淆
                if any(f"h.{i}." in n for i in range(8, 12)):  # 注意添加了点号
                    if (not decay and any(nd in n for nd in no_decay)) or (decay and not any(nd in n for nd in no_decay)):
                        params.append(p)
                        param_names_used.add(n)
                        
            if params:
                optimizer_grouped_params.append({
                    "params": params,
                    "weight_decay": self.args.weight_decay if decay else 0.0,
                    "lr": self.args.learning_rate * 0.8
                })
        
        # 3. 底部Transformer层 (h.0 到 h.7) - 使用更精确的模式匹配
        for decay in [True, False]:
            params = []
            for n, p in self.student_model.named_parameters():
                if n in param_names_used:
                    continue
                    
                # 使用更精确的模式匹配，添加点号确保精确匹配
                if any(f"h.{i}." in n for i in range(8)):  # 注意添加了点号
                    if (not decay and any(nd in n for nd in no_decay)) or (decay and not any(nd in n for nd in no_decay)):
                        params.append(p)
                        param_names_used.add(n)
                        
            if params:
                optimizer_grouped_params.append({
                    "params": params,
                    "weight_decay": self.args.weight_decay if decay else 0.0,
                    "lr": self.args.learning_rate * 0.4
                })
        
        # 4. 检查是否有未分组的参数
        remaining_params = []
        for n, p in self.student_model.named_parameters():
            if n not in param_names_used and p.requires_grad:
                remaining_params.append(p)
                logger.warning(f"Parameter not in any group: {n}")
        
        if remaining_params:
            optimizer_grouped_params.append({
                "params": remaining_params,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate * 0.1  # 给未分组参数较低学习率
            })
        
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_params,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 计算训练步数
        total_steps = len(self.train_dataloader) * self.args.num_epochs
        
        # 创建学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[g["lr"] for g in optimizer_grouped_params],  # 分组学习率
            total_steps=total_steps,
            pct_start=self.args.warmup_ratio,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        logger.info(f"Initialized optimizer with {len(optimizer_grouped_params)} parameter groups")
        logger.info(f"Training steps: {total_steps}")

    def _save_checkpoint(self, checkpoint_name, epoch):
        """Save model checkpoint"""
        # Only save from main process
        if not self.accelerator.is_main_process:
            return
            
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # Get unwrapped model
        unwrapped_model = self.accelerator.unwrap_model(self.student_model)
        
        # Save model state dict
        model_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(unwrapped_model.state_dict(), model_path)
        
        # Save training state
        training_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "args": vars(self.args),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        torch.save(training_state, state_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # Log with wandb if enabled
        if self.args.use_wandb and checkpoint_name == "best_model":
            artifact = wandb.Artifact(f"distilled_model_{self.global_step}", type="model")
            artifact.add_dir(checkpoint_dir)
            wandb.log_artifact(artifact)
            
    def _load_checkpoint(self, checkpoint_dir):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_dir}...")
        
        # Load training state
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(state_path):
            training_state = torch.load(state_path, map_location="cpu")
            self.start_epoch = training_state["epoch"] + 1
            self.global_step = training_state["global_step"]
            self.best_loss = training_state["best_loss"]
            
            # Load optimizer and scheduler states
            if "optimizer" in training_state and hasattr(self, "optimizer"):
                try:
                    self.optimizer.load_state_dict(training_state["optimizer"])
                    logger.info("Optimizer state restored")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")
                    
            if "scheduler" in training_state and hasattr(self, "lr_scheduler") and training_state["scheduler"]:
                try:
                    self.lr_scheduler.load_state_dict(training_state["scheduler"])
                    logger.info("Learning rate scheduler state restored")
                except Exception as e:
                    logger.warning(f"Failed to load scheduler state: {e}")
        
        # Load model weights
        model_path = os.path.join(checkpoint_dir, "model.pth")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            unwrapped_model = self.accelerator.unwrap_model(self.student_model)
            unwrapped_model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully")
        
        logger.info(f"Resuming from epoch {self.start_epoch}, best loss: {self.best_loss:.4f}")

    def kd_loss_fn(self, student_logits, teacher_logits, temperature=0.5, alpha=0.3, targets=None):
        """
        改进的知识蒸馏损失函数，增加归一化处理并动态调整权重
        """
        # 对logits进行归一化处理
        if self.args.normalize_logits:
            # 中心化处理
            t_mean = teacher_logits.mean(dim=-1, keepdim=True)
            s_mean = student_logits.mean(dim=-1, keepdim=True)
            
            # 可选：标准化处理，使分布更匹配
            if self.args.standardize_logits:
                t_std = torch.std(teacher_logits, dim=-1, keepdim=True) + 1e-6
                s_std = torch.std(student_logits, dim=-1, keepdim=True) + 1e-6
                norm_teacher_logits = (teacher_logits - t_mean) / t_std
                norm_student_logits = (student_logits - s_mean) / s_std
            else:
                # 只进行中心化
                norm_teacher_logits = teacher_logits - t_mean
                norm_student_logits = student_logits - s_mean
        else:
            norm_teacher_logits = teacher_logits
            norm_student_logits = student_logits
        
        # 计算软目标损失（KL散度）
        soft_targets = F.softmax(norm_teacher_logits / temperature, dim=-1)
        log_probs = F.log_softmax(norm_student_logits / temperature, dim=-1)
        
        soft_targets_loss = F.kl_div(
            log_probs, 
            soft_targets, 
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # 硬目标损失
        hard_targets_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1)
        )
        
        if self.args.dynamic_temperature:
            # 随训练进度降低温度，更聚焦于硬标签
            total_steps = self.args.num_epochs * len(self.train_dataloader)
            progress = min(1.0, self.global_step / (total_steps * 0.8))
            
            # 从初始温度下降到一个更低的值
            current_temperature = temperature * max(0.5, 1.0 - progress * 0.5)
        else:
            current_temperature = temperature

        # 使用current_temperature替换temperature
        soft_targets = F.softmax(norm_teacher_logits / current_temperature, dim=-1)
        log_probs = F.log_softmax(norm_student_logits / current_temperature, dim=-1)

        # 随着训练进行，逐渐降低KL损失权重，增加CE损失权重
        if self.args.dynamic_alpha:
            total_steps = self.args.num_epochs * len(self.train_dataloader)
            progress = min(1.0, self.global_step / (total_steps * 0.8))
            
            # 使用更激进的衰减曲线
            if progress < 0.4:
                # 前40%缓慢降低
                current_alpha = alpha * (1.0 - progress * 0.4)  # 更大的系数
            else:
                # 后60%快速降低至更小值
                current_alpha = alpha * max(0.1, 1.0 - 0.2 - (progress-0.4) * 1.5)  # 降至原始alpha的10%
            
            # 第3阶段特殊处理 - 更强调CE loss
            if self.args.use_progressive_training:
                stage3_start = self.args.progressive_stage2_epochs * len(self.train_dataloader)
                
                # 第3阶段的前5个epoch特别强调CE loss
                stage3_special = stage3_start + 5 * len(self.train_dataloader)
                
                if stage3_start <= self.global_step <= stage3_special:
                    # 在第三阶段开始时大幅降低alpha (降至正常值的50%)
                    current_alpha = current_alpha * 0.5
                    
                    # 每30步完全使用CE loss训练一个批次
                    if self.global_step % 30 == 0:
                        current_alpha = 0.0  # 完全使用CE loss
        else:
            current_alpha = alpha
                    
        # 组合损失
        total_loss = (current_alpha * soft_targets_loss) + ((1 - current_alpha) * hard_targets_loss)
        
            
        # 只在主进程记录日志，并使用synchronize_step=False避免步数检查
        if self.global_step % 50 == 0 and self.accelerator.is_main_process:
            logger.info(f"Step {self.global_step}: "
                    f"Total Loss={total_loss.item():.4f}, "
                    f"KL Loss={soft_targets_loss.item():.4f}, "
                    f"CE Loss={hard_targets_loss.item():.4f}")
            
            # 添加logits范围检查，帮助分析问题
            t_min, t_max = teacher_logits.min().item(), teacher_logits.max().item()
            s_min, s_max = student_logits.min().item(), student_logits.max().item()
            logger.info(f"Logits ranges - Teacher: [{t_min:.2f}, {t_max:.2f}], Student: [{s_min:.2f}, {s_max:.2f}]")
            
            # 记录到wandb但禁用步数检查
            if self.args.use_wandb:
                wandb.log({
                    "soft_loss": soft_targets_loss.item(),
                    "hard_loss": hard_targets_loss.item(),
                    "total_loss": total_loss.item(),
                    "teacher_logits_range": [t_min, t_max],
                    "student_logits_range": [s_min, s_max],
                }, step=self.global_step, commit=True)
        
        return total_loss
        
    def train(self):
        """Run full training loop"""
        logger.info("Starting training loop...")
        
        # Track metrics
        train_losses = []
        
        # Training loop
        for epoch in range(self.start_epoch, self.args.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            if self.args.use_progressive_training:
                self._progressive_training(epoch)

            # Train for one epoch
            train_loss = self._train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
                       
            # Log to wandb
            if self.args.use_wandb and self.accelerator.is_main_process:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch_loss": train_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint if best loss
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self._save_checkpoint("best_model", epoch)
                logger.info(f"New best model saved, loss: {train_loss:.4f}")
            
            # Always save latest checkpoint
            self._save_checkpoint(f"checkpoint_epoch_{epoch+1}", epoch)
            
        logger.info(f"Training complete. Best loss: {self.best_loss:.4f}")
        
        # Final save
        self._save_checkpoint("final_model", self.args.num_epochs-1)
        
        # End wandb run
        if self.args.use_wandb and self.accelerator.is_main_process:
            # Log final results
            wandb.log({"best_loss": self.best_loss})
            # Upload final model
            artifact = wandb.Artifact(f"final_distilled_model", type="model")
            artifact.add_dir(os.path.join(self.args.output_dir, "final_model"))
            wandb.log_artifact(artifact)
            # End tracking
            self.accelerator.end_training()
        
        return self.best_loss
    
    def _train_epoch(self, epoch):
        """Train for one epoch with gradient accumulation"""
        self.student_model.train()
        self.teacher_model.eval()  # Teacher always in eval mode
        total_loss = 0.0
        
        # Use tqdm for progress display (only on main process)
        if self.accelerator.is_main_process:
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
        else:
            pbar = self.train_dataloader
        
        # For gradient accumulation
        self.optimizer.zero_grad()
        accumulated_steps = 0
        
        for step, patches in enumerate(pbar):
            # Ensure batch is on correct device
            batch_patches = patches.to(self.device, non_blocking=True)
            b, n, c, h, w = batch_patches.shape
            input_patches = batch_patches.view(b*n, c, h, w)
            
            # Forward pass through teacher model (no gradients)
            with torch.no_grad():
                teacher_logits, _ = self.teacher_model(input_patches, None)
            
            # Forward pass through student model
            with self.accelerator.autocast():
                student_logits, _ = self.student_model(input_patches, None)
                
                # Get pixel IDs for hard targets
                pixel_ids = input_patches.long().clamp(0, 255).view(b*n, c*h*w)
                
                # Shift for autoregressive prediction
                shifted_teacher_logits = teacher_logits[:, :-1].contiguous()
                shifted_student_logits = student_logits[:, :-1].contiguous()
                shifted_targets = pixel_ids[:, 1:].contiguous()
                
                # Calculate distillation loss
                loss = self.kd_loss_fn(
                    shifted_student_logits, 
                    shifted_teacher_logits,
                    temperature=self.args.temperature,
                    alpha=self.args.alpha,
                    targets=shifted_targets
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.args.gradient_accumulation_steps
            
            # Backward pass
            self.accelerator.backward(loss)
            accumulated_steps += 1
            
            # Update weights after accumulating enough gradients
            if accumulated_steps == self.args.gradient_accumulation_steps:
                has_nan_grad = False
                for name, param in self.student_model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        if self.accelerator.is_main_process:
                            logger.warning(f"NaN/Inf gradients detected in {name}")
                            break
                
                if has_nan_grad:
                    logger.warning("Skipping update due to NaN/Inf gradients")
                    self.optimizer.zero_grad()
                    accumulated_steps = 0
                    continue

                # Gradient clipping
                self.accelerator.clip_grad_norm_(
                    self.student_model.parameters(), 
                    self.args.max_grad_norm
                )
                
                # Update weights
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                accumulated_steps = 0
                
                # Update global step (only after actual weight update)
                self.global_step += 1
                
                # Save checkpoint every N steps
                if self.global_step % self.args.save_steps == 0:
                    self._save_checkpoint(f"checkpoint_step_{self.global_step}", epoch)
                    logger.info(f"Step checkpoint saved (step {self.global_step})")
            
            # Update progress bar
            loss_item = loss.detach().float() * self.args.gradient_accumulation_steps  # Rescale for logging
            loss_item = self.accelerator.gather(loss_item).mean().item()
            total_loss += loss_item
            avg_loss = total_loss / (step + 1)
            
            # Update progress bar on main process
            if self.accelerator.is_main_process:
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # Log metrics to wandb
                if self.args.use_wandb and step % self.args.wandb_log_steps == 0:
                    wandb.log({
                        "step": self.global_step,
                        "step_loss": loss_item,
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch_progress": epoch + (step / len(self.train_dataloader)),
                        "avg_loss": avg_loss,
                        "gpu_memory": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    })
            
        
        # Ensure all processes finish and loss values are consistent
        avg_loss = total_loss / len(self.train_dataloader)
        return self.accelerator.gather(torch.tensor(avg_loss, device=self.accelerator.device)).mean().item()
    
    def _progressive_training(self, epoch):
        """实现渐进式训练策略"""
        if not self.args.use_progressive_training:
            return
            
        # 第一阶段：仅训练嵌入和输出层（前几个epoch）
        if epoch < self.args.progressive_stage1_epochs:
            logger.info("Progressive Training Stage 1: Training embedding and output layers only")
            for name, param in self.student_model.named_parameters():
                if any(nd in name for nd in ["wte", "prob"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
        # 第二阶段：加入顶层Transformer层
        elif epoch < self.args.progressive_stage2_epochs:
            logger.info("Progressive Training Stage 2: Adding top transformer layers")
            for name, param in self.student_model.named_parameters():
                if (any(nd in name for nd in ["wte", "prob"]) or 
                    any(f"h.{i}" in name for i in range(8, 12))):  # 假设12层，解冻最后4层
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        # 第三阶段：训练所有层
        else:
            logger.info("Progressive Training Stage 3: Training all parameters with adjusted learning rates")
            # 解冻所有参数
            for param in self.student_model.parameters():
                param.requires_grad = True
            
            # 仅在刚进入第三阶段时重置优化器和调整学习率
            if epoch == self.args.progressive_stage2_epochs:
                # 保存原始学习率
                old_lr = self.args.learning_rate
                
                # 临时提高学习率用于初始化优化器
                self.args.learning_rate = old_lr * 2.5
                
                # 调整层级学习率系数 - 特别提高底层Transformer的学习率
                self._init_optimizer_with_custom_lrs(
                    embed_lr_factor=0.8,    # 降低嵌入层，因为已经训练了两个阶段
                    top_lr_factor=1.2,      # 保持顶层不变
                    bottom_lr_factor=2.0    # 大幅提高底层学习率
                )
                
                # 恢复原始学习率参数但不再使用
                self.args.learning_rate = old_lr
                
                # 准备优化器
                self.optimizer = self.accelerator.prepare(self.optimizer)
                
                logger.info("Stage 3: Enhanced optimizer with boosted learning rates for bottom layers")

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Knowledge distillation from LLaMA to GPT2 for image compression")
    parser.add_argument("--dynamic_temperature", action="store_true", default=True,
                    help="Dynamically decrease temperature during training")
    # Model paths
    parser.add_argument("--teacher_model_path", type=str, default="/remote-home/wufeiyang/final_model.pth",
                    help="Path to teacher model checkpoint")
    parser.add_argument("--llama_model_dir", type=str, default="/remote-home/wufeiyang/saved_model",
                    help="Path to LLaMA model configuration directory")
    parser.add_argument("--gpt2_model_dir", type=str, default="/remote-home/wufeiyang/gpt2_model",
                    help="Path to GPT2 model")
    parser.add_argument("--use_lora_teacher", action="store_true", default=True,
                help="Use LoRA-enhanced teacher model")
    parser.add_argument("--lora_path", type=str, default="/remote-home/wufeiyang/best_model",
                    help="Path to LoRA adapter weights")
    
    # Dataset parameters
    parser.add_argument("--train_dataset", type=str, default="/remote-home/wufeiyang/dataset/flickr30k/Images",
                    help="Path to training dataset")
    parser.add_argument("--patch_size", type=int, default=16,
                    help="Size of image patches")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=24,
                    help="Training batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=50,
                    help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                    help="Peak learning rate")
    parser.add_argument("--transformer_lr_factor", type=float, default=0.5,
                    help="Factor for transformer layers learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                    help="Weight decay for AdamW")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                    help="Proportion of steps for learning rate warmup")
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                    help="Maximum gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,  # New parameter
                    help="Number of updates steps to accumulate before performing backward/update")
    parser.add_argument("--num_workers", type=int, default=2,
                    help="Number of data loader workers")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Resume training from checkpoint")
    
    # 蒸馏参数
    parser.add_argument("--temperature", type=float, default=1.0,  # 降低温度
                    help="Temperature for knowledge distillation")
    parser.add_argument("--alpha", type=float, default=0.7,  # 降低KL散度权重
                    help="Weight for soft targets (1-alpha for hard targets)")
    parser.add_argument("--normalize_logits", action="store_true", default=True,
                    help="Normalize logits before computing distillation loss")

    # 渐进式训练参数
    parser.add_argument("--use_progressive_training", action="store_true", default=True,
                    help="Use progressive layer unfreezing during training")
    
    # Checkpointing parameters
    parser.add_argument("--save_steps", type=int, default=5000,
                    help="Save checkpoint every N steps")
    
    # Wandb parameters
    parser.add_argument("--use_wandb", action="store_true", default=True,
                    help="Use wandb for training monitoring")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                    help="Wandb run name")
    parser.add_argument("--wandb_log_steps", type=int, default=10,
                    help="Log to wandb every N steps")
    
    # Accelerator parameters
    parser.add_argument("--use_deepspeed", action="store_true", default=True,
                    help="Use DeepSpeed plugin with Accelerator")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./distilled_model",
                    help="Directory to save output files")
    
    parser.add_argument("--progressive_stage1_epochs", type=int, default=2,
                    help="Number of epochs for stage 1 (embedding and output only)")
    parser.add_argument("--progressive_stage2_epochs", type=int, default=8,
                    help="Number of epochs for stage 2 (top layers)")
    parser.add_argument("--standardize_logits", action="store_true", default=True,
                    help="Standardize logits (z-score) before distillation")
    parser.add_argument("--dynamic_alpha", action="store_true", default=True,
                    help="Dynamically decrease alpha during training")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Allow flexible memory allocation for large-scale training
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Parse arguments
    args = get_args()
    
    # Create and run trainer
    trainer = DistillationTrainer(args)
    best_loss = trainer.train()
    
    logger.info(f"Training complete. Final best loss: {best_loss:.4f}")
    
    return best_loss

if __name__ == "__main__":
    main()
