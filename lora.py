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

from accelerate import Accelerator, DeepSpeedPlugin
from peft import LoraConfig, get_peft_model, PeftModel
from main import SafeLlamaPixelAR
from torch.utils.data.distributed import DistributedSampler

# CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 lora.py \
#   --batch_size 24 \
#   --per_device_batch_size 8 \
#   --gradient_checkpointing

ImageFile.LOAD_TRUNCATED_IMAGES = True

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
local_rank = int(os.environ.get("LOCAL_RANK", -1))
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d-%H%M%S')}_rank{local_rank}.log")

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

def init_accelerator(args):
    if args.local_rank != -1:
        return Accelerator(
            mixed_precision="bf16",
            log_with="wandb" if args.use_wandb else None,
            deepspeed_plugin=DeepSpeedPlugin(
                zero_stage=2,
                offload_optimizer_device="none"
            )
        )
    else:
        return Accelerator(
            mixed_precision="bf16",
            log_with="wandb" if args.use_wandb else None
        )

class RobustImageDataset(Dataset):
    def __init__(self, root_dir, size=256, patch_size=16, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.size = size
        self.patch_size = patch_size
        self._init_paths()
        if transform is None:
            self.transform = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Lambda(lambda x: x * 255)
            ])
        else:
            self.transform = transform

    def _init_paths(self):
        self.paths = sorted(glob.glob(os.path.join(self.root_dir, "*.jpg")))
        logger.info(f"找到 {len(self.paths)} 个图像文件")

    def __len__(self):
        return len(self.paths)

    def random_patchify(self, img_tensor):
        import random
        c, h, w = img_tensor.shape
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        patch = img_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
        return patch

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img)
            patch = self.random_patchify(img_tensor)
            pixel_values = patch.long().clamp(0, 255)
            b, h, w = pixel_values.shape
            seq_len = b * h * w
            pixel_ids = pixel_values.reshape(seq_len)
            input_ids = pixel_ids[:-1]
            labels = pixel_ids[1:]
            return {
                "input_ids": input_ids, 
                "labels": labels,
                "pixel_values": pixel_values,
            }
        except Exception as e:
            logger.warning(f"加载图像 {path} 出错: {str(e)}，跳过...")
            return self[(idx + 1) % len(self)]

class LoraFineTuner:
    def __init__(self, args):
        self.args = args
        self.accelerator = init_accelerator(args)
        if args.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                entity="feiyangwu0309-sjtu-icec",
                project="llama-compression", 
                name=args.wandb_run_name or f"lora-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(args)
            )
            wandb.run.notes = f"使用LoRA微调LLaMA压缩模型，秩={args.lora_rank}"
            logger.info(f"已初始化wandb，项目：llama-compression")

        if args.use_wandb:
            self.accelerator.init_trackers("llama-compression")
        self.device = self.accelerator.device
        logger.info(f"使用设备: {self.device}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if self.accelerator.is_main_process and not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        self._init_model()
        self._init_dataset()
        self._init_optimizer()
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader
        )
        self.base_model = self.base_model.to(self.accelerator.device)
        self.base_model = self.base_model.to(dtype=torch.bfloat16)
        self.base_model.prob = self.base_model.prob.to(self.accelerator.device)
        self.base_model.prob = self.base_model.prob.to(dtype=torch.bfloat16)
        self.global_step = 0
        self.best_loss = float("inf")
        self.start_epoch = 0
        self.last_checkpoint_time = time.time()
        if args.resume_from_checkpoint:
            self._load_checkpoint(args.resume_from_checkpoint)

    def _init_model(self):
        logger.info("正在初始化模型...")
        self.base_model = SafeLlamaPixelAR(model_path=self.args.model_dir)
        self.base_model = self.base_model.to(self.accelerator.device)
        self.base_model = self.base_model.to(dtype=torch.bfloat16)
        checkpoint = torch.load(self.args.model_checkpoint, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if all(k.startswith("module.") for k in checkpoint.keys()):
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("llama.model."):
                new_key = key.replace("llama.model.", "llama.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        self.base_model.load_state_dict(new_state_dict, strict=False)
        logger.info("基础模型加载成功")
        self.llama_model = self.base_model.llama
        logger.info("应用LoRA配置...")
        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "down_proj", "up_proj"
            ]
        )
        self.model = get_peft_model(self.llama_model, lora_config)
        trainable_params = self.model.print_trainable_parameters()
        logger.info(f"创建的LoRA模型: {trainable_params}")
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def _init_dataset(self):
        logger.info("加载数据集...")
        self.train_dataset = RobustImageDataset(
            root_dir=self.args.train_dataset,
            size=256,
            patch_size=16
        )
        logger.info(f"训练数据集: {len(self.train_dataset)} 图像")
        train_sampler = DistributedSampler(self.train_dataset) if self.accelerator.num_processes > 1 else None
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_batch_size,  # 不是batch_size
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

    def _init_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999)
        )
        train_steps = len(self.train_dataloader) * self.args.num_epochs
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
        logger.info("开始训练循环...")
        train_losses = []
        for epoch in range(self.start_epoch, self.args.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.args.num_epochs}")
            train_loss = self._train_epoch(epoch)
            train_losses.append(train_loss)
            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
            if self.args.use_wandb and self.accelerator.is_main_process:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch_loss": train_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self._save_checkpoint("best_model", epoch)
                logger.info(f"已保存新的最佳模型，损失值: {train_loss:.4f}")
            self._save_checkpoint(f"checkpoint_epoch_{epoch+1}", epoch)
        logger.info(f"训练完成。最佳损失: {self.best_loss:.4f}")
        self._save_checkpoint("final_model", self.args.num_epochs-1)
        if self.args.use_wandb and self.accelerator.is_main_process:
            wandb.log({"best_loss": self.best_loss})
            artifact = wandb.Artifact(f"lora_model", type="model")
            artifact.add_dir(os.path.join(self.args.output_dir, "best_model"))
            wandb.log_artifact(artifact)
            self.accelerator.end_training()
        return self.best_loss

    def _train_epoch(self, epoch):
        if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(epoch)  
        self.model.train()
        total_loss = 0.0
        checkpoint_interval = self.args.checkpoint_interval * 60
        if self.accelerator.is_main_process:
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
        else:
            pbar = self.train_dataloader
        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with self.accelerator.autocast():
                outputs = self.model(input_ids=batch["input_ids"])
                hidden_states = outputs.last_hidden_state
                logits = self.base_model.prob(hidden_states)
                loss = self.base_model.loss_fn(
                    logits.view(-1, 256),
                    batch["labels"].view(-1)
                )
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            loss_item = loss.detach().float()
            loss_item = self.accelerator.gather(loss_item).mean().item()
            total_loss += loss_item
            avg_loss = total_loss / (step + 1)
            if self.accelerator.is_main_process:
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                if self.args.use_wandb and step % self.args.wandb_log_steps == 0:
                    wandb.log({
                        "step": self.global_step,
                        "step_loss": loss_item,
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch_progress": epoch + (step / len(self.train_dataloader)),
                        "avg_loss": avg_loss,
                        "gpu_memory": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    })
                    if self.global_step % 500 == 0:
                        tparams = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                        if tparams:
                            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in tparams]))
                            wandb.log({"gradient_norm": grad_norm}, step=self.global_step)
            self.global_step += 1
            current_time = time.time()
            if current_time - self.last_checkpoint_time > checkpoint_interval:
                self._save_checkpoint(f"checkpoint_time_{int(current_time)}", epoch)
                logger.info(f"已保存时间检查点 (step {self.global_step})")
                self.last_checkpoint_time = current_time
            if self.global_step % self.args.save_steps == 0:
                self._save_checkpoint(f"checkpoint_step_{self.global_step}", epoch)
                logger.info(f"已保存步骤检查点 (step {self.global_step})")
        avg_loss = total_loss / len(self.train_dataloader)
        return self.accelerator.gather(torch.tensor(avg_loss, device=self.accelerator.device)).mean().item()

    def _save_checkpoint(self, checkpoint_name, epoch):
        if not self.accelerator.is_main_process:
            return
        checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save
        )
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
        if self.args.use_wandb and checkpoint_name == "best_model":
            artifact = wandb.Artifact(f"lora_model_{self.global_step}", type="model")
            artifact.add_dir(checkpoint_dir)
            wandb.log_artifact(artifact)

    def _load_checkpoint(self, checkpoint_dir):
        logger.info(f"正在从 {checkpoint_dir} 加载检查点...")
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(state_path):
            training_state = torch.load(state_path, map_location="cpu")
            self.start_epoch = training_state["epoch"] + 1
            self.global_step = training_state["global_step"]
            self.best_loss = training_state["best_loss"]
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
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model = PeftModel.from_pretrained(
            unwrapped_model,
            checkpoint_dir,
            is_trainable=True
        )
        self.model = self.accelerator.prepare(unwrapped_model) 
        self.last_checkpoint_time = time.time()
        logger.info(f"从epoch {self.start_epoch}加载的检查点，loss {self.best_loss:.4f}")

def get_args():
    parser = argparse.ArgumentParser(description="使用LoRA为基于LLaMA的图像压缩模型进行微调")
    parser.add_argument("--model_checkpoint", type=str, default="/remote-home/wufeiyang/final_model.pth")
    parser.add_argument("--model_dir", type=str, default="/remote-home/wufeiyang/saved_model")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--train_dataset", type=str, default="/remote-home/wufeiyang/dataset/flickr30k/Images")
    parser.add_argument("--patch_size", type=int, default=16)
    # parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=60)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_log_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./lora_output")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    return parser.parse_args()

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = get_args()
    fine_tuner = LoraFineTuner(args)
    best_loss = fine_tuner.train()
    logger.info(f"训练完成。最终最佳损失: {best_loss:.4f}")
    return best_loss

if __name__ == "__main__":
    main()