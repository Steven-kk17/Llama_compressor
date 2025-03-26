import os
import torch
import torch.nn as nn
import argparse
import logging
import glob
import numpy as np
import math
from tqdm import tqdm

from PIL import Image, ImageFile
from peft import PeftModel

from test import ModelTester
from gpt2 import GPT2PixelAR

# 允许加载部分损坏/截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 创建结果目录
os.makedirs("./gpt2_lora_results", exist_ok=True)

# 定义DirectoryImageDataset类
class DirectoryImageDataset:
    """简单的图像目录数据集类"""
    def __init__(self, image_files):
        self.image_files = image_files
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_files[idx]).convert('RGB')
            return {"image": img, "file": self.image_files[idx]}
        except Exception as e:
            logger.error(f"Error loading image {self.image_files[idx]}: {e}")
            # 返回一个小的黑色图像作为替代
            return {"image": Image.new('RGB', (256, 256), color='black'), 
                    "file": self.image_files[idx]}

class GPT2LoRATester(ModelTester):
    def __init__(self, model_path="./distilled_model/best_model/model.pth", dataset=None,
                 gpt2_model_dir="/remote-home/wufeiyang/gpt2_model",
                 skip_ac=True, lora_path="./gpt2_lora_output/best_model", save_images=False,
                 keep_original_size=True, batch_size=8, single_gpu=False):
        """
        初始化GPT2 LoRA测试器，继承自ModelTester
        
        Args:
            model_path: 蒸馏模型路径
            dataset: 数据集
            gpt2_model_dir: GPT2模型目录
            skip_ac: 是否跳过算术编码
            lora_path: LoRA权重路径
            save_images: 是否保存图像
            keep_original_size: 是否保持原始尺寸（默认为True，与distillation_test保持一致）
            batch_size: 批处理大小
            single_gpu: 是否只使用单GPU
        """
        self.gpt2_model_dir = gpt2_model_dir
        self.lora_path = lora_path
        self.single_gpu = single_gpu
        self.results_dir = "./gpt2_lora_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.batch_size = batch_size
        
        # 总是保持原始尺寸，与distillation_test保持一致
        keep_original_size = True
        
        # 调用父类的初始化方法
        super().__init__(model_path=model_path, dataset=dataset, 
                         saved_model_dir=gpt2_model_dir, skip_ac=skip_ac, 
                         save_images=save_images, keep_original_size=keep_original_size,
                         batch_size=batch_size)
        
    def _init_model(self):
        """重写初始化模型方法，适配GPT2+LoRA"""
        logger.info(f"从 {self.model_path} 加载GPT2模型并应用LoRA权重")
        
        # 加载GPT2基础模型
        self.base_model = GPT2PixelAR(model_path=self.gpt2_model_dir)
        
        # 加载蒸馏检查点
        distil_checkpoint = torch.load(self.model_path, map_location="cpu")
        
        # 处理可能的state_dict格式
        if isinstance(distil_checkpoint, dict) and "state_dict" in distil_checkpoint:
            distil_checkpoint = distil_checkpoint["state_dict"]
            
        # 处理可能的module前缀
        if all(k.startswith("module.") for k in distil_checkpoint.keys()):
            distil_checkpoint = {k[7:]: v for k, v in distil_checkpoint.items()}
                
        # 加载状态字典到基础模型
        self.base_model.load_state_dict(distil_checkpoint, strict=False)
        logger.info("基础模型加载成功")
        
        # 加载prob层权重
        prob_weights_path = os.path.join(self.lora_path, "prob_layer.pt")
        if os.path.exists(prob_weights_path):
            prob_weights = torch.load(prob_weights_path, map_location="cpu")
            self.base_model.prob.load_state_dict(prob_weights)
            logger.info("成功加载Prob层权重")
        
        # 加载LoRA权重
        logger.info(f"从 {self.lora_path} 加载LoRA适配器")
        gpt2_model = self.base_model.gpt2
        self.model = PeftModel.from_pretrained(
            gpt2_model, 
            self.lora_path,
            is_trainable=False  # 只进行推理，不训练
        )
        
        # 合并LoRA权重到基础模型
        logger.info("合并LoRA权重到基础模型中以加速推理...")
        self.model = self.model.merge_and_unload()
        logger.info("LoRA权重合并成功")
        
        # 检查可用GPU并使用DataParallel
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1 and not self.single_gpu:
            logger.info(f"使用 {num_gpus} 个GPU进行推理")
            self.model = nn.DataParallel(self.model)
        else:
            if num_gpus > 1:
                logger.info(f"强制使用单GPU模式 (忽略额外的 {num_gpus-1} 个GPU)")
            else:
                logger.info(f"使用单GPU进行推理")
        
        # 将模型移至正确的设备
        self.model = self.model.to(self.device).eval()
        self.base_model.prob = self.base_model.prob.to(self.device).eval()
        logger.info("模型初始化完成")

    def process_dataset_image(self, img):
        """重写处理图像方法来正确处理GPT2+LoRA的输入和输出，保持与distillation_test一致"""
        # 应用变换但不调整尺寸
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        b, c, h, w = img_tensor.shape
        
        # 保存原始尺寸信息，用于后面的掩码和计算
        self.original_h, self.original_w = h, w
        
        # 根据patch_size计算需要的填充量以确保可被整除
        pad_h = 0 if h % self.patch_size == 0 else self.patch_size - (h % self.patch_size)
        pad_w = 0 if w % self.patch_size == 0 else self.patch_size - (w % self.patch_size)
        
        if pad_h > 0 or pad_w > 0:
            logger.info(f"填充图像从 {h}x{w} 到 {h+pad_h}x{w+pad_w} 以被patch大小 {self.patch_size} 整除")
            # 只在右侧和底部填充
            padder = nn.ZeroPad2d((0, pad_w, 0, pad_h))
            img_tensor = padder(img_tensor)
            
            # 创建掩码以跟踪原始像素
            self.pixel_mask = torch.ones((b, c, h+pad_h, w+pad_w), dtype=torch.bool, device=self.device)
            self.pixel_mask[:, :, h:, :] = False  # 底部填充区域标记为False
            self.pixel_mask[:, :, :, w:] = False  # 右侧填充区域标记为False
        else:
            # 如果不需要填充，全部是原始像素
            self.pixel_mask = torch.ones_like(img_tensor, dtype=torch.bool)
        
        # 更新尺寸（可能已经填充）
        b, c, h, w = img_tensor.shape
        
        # 确保可以被patch_size整除
        assert h % self.patch_size == 0 and w % self.patch_size == 0, f"图像尺寸 {h}x{w} 不能被patch大小 {self.patch_size} 整除"
        
        # 计算需要多少个patch
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        total_patches = num_patches_h * num_patches_w
        
        # 设置最大区域大小 (以patch为单位)，不超过16x16个patch
        max_region_size = 16  # patches
        
        logger.info(f"处理图像: {b}x{c}x{h}x{w}, 分割成 {num_patches_h}x{num_patches_w} 个patches")
        
        # 检查是否需要分块处理
        if num_patches_h > max_region_size or num_patches_w > max_region_size:
            logger.info(f"检测到大图像，以最大 {max_region_size}x{max_region_size} patches的区域进行处理")
            
            # 分区域处理
            all_logits = []
            total_bpp = 0.0
            
            # 计算需要的区域数量
            num_regions_h = math.ceil(num_patches_h / max_region_size)
            num_regions_w = math.ceil(num_patches_w / max_region_size)
            total_regions = num_regions_h * num_regions_w
            
            logger.info(f"图像将分 {total_regions} 个区域处理")
            
            # 获取可用 GPU 数量
            num_gpus = torch.cuda.device_count() if not self.single_gpu else 1
            if num_gpus > 1:
                logger.info(f"使用 {num_gpus} 个GPU进行区域处理")
            
            # 逐区域处理
            region_count = 0
            for region_i in range(num_regions_h):
                for region_j in range(num_regions_w):
                    # GPU 选择逻辑
                    if num_gpus > 1:
                        gpu_id = region_count % num_gpus
                        current_device = torch.device(f"cuda:{gpu_id}")
                        logger.info(f"处理区域 ({region_i},{region_j}) 在GPU {gpu_id}上")
                    else:
                        current_device = self.device
                    
                    region_count += 1
                    
                    # 计算当前区域的patch范围
                    start_h = region_i * max_region_size
                    start_w = region_j * max_region_size
                    end_h = min(start_h + max_region_size, num_patches_h)
                    end_w = min(start_w + max_region_size, num_patches_w)
                    
                    region_patches_h = end_h - start_h
                    region_patches_w = end_w - start_w
                    
                    logger.info(f"处理区域 ({region_i},{region_j}): {region_patches_h}x{region_patches_w} patches")
                    
                    # 提取当前区域
                    region_img = img_tensor[:, :, 
                                    start_h*self.patch_size:end_h*self.patch_size,
                                    start_w*self.patch_size:end_w*self.patch_size]
                    
                    if num_gpus > 1:
                        region_img = region_img.to(current_device)
                    
                    # 当前区域的所有patches
                    patches = []
                    for i in range(region_patches_h):
                        for j in range(region_patches_w):
                            patch = region_img[:, :, 
                                        i*self.patch_size:(i+1)*self.patch_size, 
                                        j*self.patch_size:(j+1)*self.patch_size]
                            patches.append(patch)
                    
                    # 分批处理当前区域的patches
                    region_logits = []
                    region_bpp = 0.0
                    
                    # GPT2+LoRA区域处理
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            # 批量处理patches
                            for i in range(0, len(patches), self.batch_size):
                                batch_patches = torch.cat(patches[i:min(i+self.batch_size, len(patches))], dim=0)
                                
                                if num_gpus > 1:
                                    batch_patches = batch_patches.to(current_device)
                                    
                                current_batch_size = batch_patches.size(0)
                                
                                # 使用model_forward处理批次
                                batch_logits, batch_bpp = self.model_forward(batch_patches)
                                
                                # 处理BPP - 匹配distillation_test的更健壮处理
                                if isinstance(batch_bpp, (float, int)):
                                    region_bpp += batch_bpp * current_batch_size
                                elif isinstance(batch_bpp, torch.Tensor) and batch_bpp.numel() == 1:
                                    region_bpp += batch_bpp.item() * current_batch_size
                                elif isinstance(batch_bpp, torch.Tensor):
                                    if len(batch_bpp) == current_batch_size:
                                        # 每个图像一个BPP值 - 求和
                                        region_bpp += batch_bpp.sum().item()
                                    elif batch_bpp.dim() == 0:
                                        # 单个标量张量
                                        region_bpp += batch_bpp.item() * current_batch_size
                                    elif len(batch_bpp) == torch.cuda.device_count():
                                        # 每个GPU返回了其批次部分的平均BPP值
                                        # 需要根据每个GPU上的样本数量对这些值进行加权
                                        gpu_count = torch.cuda.device_count()
                                        
                                        # 计算每个GPU上有多少样本
                                        base_samples = current_batch_size // gpu_count
                                        extra = current_batch_size % gpu_count
                                        samples_per_gpu = [base_samples + (1 if i < extra else 0) for i in range(gpu_count)]
                                        
                                        # 按处理的样本对BPP加权
                                        weighted_bpp = sum(bpp.item() * count for bpp, count in zip(batch_bpp, samples_per_gpu))
                                        region_bpp += weighted_bpp
                                    else:
                                        # 未知格式 - 使用平均值并显示警告
                                        logger.warning(f"意外的BPP张量形状 {batch_bpp.shape}")
                                        region_bpp += batch_bpp.mean().item() * current_batch_size
                                
                                # 收集结果
                                if num_gpus > 1:
                                    chunked_logits = [logit.cpu() for logit in batch_logits.chunk(current_batch_size, dim=0)]
                                    region_logits.extend(chunked_logits)
                                else:
                                    region_logits.extend(batch_logits.chunk(current_batch_size, dim=0))
                                
                                # 清理内存
                                del batch_logits
                                torch.cuda.empty_cache()
                    
                    # 添加当前区域的结果
                    all_logits.extend(region_logits)
                    total_bpp += region_bpp
                    
                    # 清理内存
                    torch.cuda.empty_cache()
        else:
            # 图像足够小，整体处理
            # 收集所有patches
            patches = []
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    patch = img_tensor[:, :, 
                                i*self.patch_size:(i+1)*self.patch_size, 
                                j*self.patch_size:(j+1)*self.patch_size]
                    patches.append(patch)
            
            # 分批处理patches
            all_logits = []
            total_bpp = 0.0
            
            # GPT2+LoRA小图像处理
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # 批量处理patches
                    for i in range(0, len(patches), self.batch_size):
                        batch_patches = torch.cat(patches[i:min(i+self.batch_size, len(patches))], dim=0)
                        current_batch_size = batch_patches.size(0)
                        
                        # 使用model_forward处理批次
                        batch_logits, batch_bpp = self.model_forward(batch_patches)
                        
                        # 更健壮的BPP处理，与distillation_test一致
                        if isinstance(batch_bpp, (float, int)):
                            # 如果是单个值，应用到所有patches
                            total_bpp += batch_bpp * current_batch_size
                        elif isinstance(batch_bpp, torch.Tensor) and batch_bpp.numel() == 1:
                            # 单元素张量
                            total_bpp += batch_bpp.item() * current_batch_size
                        elif isinstance(batch_bpp, torch.Tensor):
                            # 如果是张量，检查长度是否匹配
                            if len(batch_bpp) == current_batch_size:
                                total_bpp += batch_bpp.sum().item()
                            elif len(batch_bpp) == torch.cuda.device_count():
                                # 每个GPU返回了其批次部分的平均BPP值
                                gpu_count = torch.cuda.device_count()
                                
                                # 计算每个GPU上有多少样本
                                base_samples = current_batch_size // gpu_count
                                extra = current_batch_size % gpu_count
                                samples_per_gpu = [base_samples + (1 if i < extra else 0) for i in range(gpu_count)]
                                
                                # 按处理的样本对BPP加权
                                weighted_bpp = sum(bpp.item() * count for bpp, count in zip(batch_bpp, samples_per_gpu))
                                total_bpp += weighted_bpp
                            else:
                                # 其他长度不匹配情况
                                total_bpp += batch_bpp[0].item() * current_batch_size
                                logger.warning(f"警告: BPP张量长度 ({len(batch_bpp)}) 与批次大小不匹配 ({current_batch_size})")
                        else:
                            # 其他情况，使用batch大小乘以BPP
                            logger.warning(f"警告: 意外的BPP类型: {type(batch_bpp)}")
                            total_bpp += float(batch_bpp) * current_batch_size
                        
                        # 收集结果
                        if self.single_gpu:
                            all_logits.extend(batch_logits.chunk(current_batch_size, dim=0))
                        else:
                            chunked_logits = [logit.cpu() for logit in batch_logits.chunk(current_batch_size, dim=0)]
                            all_logits.extend(chunked_logits)
                        
                        # 清理内存
                        del batch_logits
                        torch.cuda.empty_cache()
        
        # 计算平均BPP
        avg_bpp = total_bpp / total_patches
        
        # 与distillation_test保持一致，避免内存错误
        logger.info("跳过完整logits合并以节省内存 (仅需计算BPP)")
        
        # 选择第一个logit作为占位符，与distillation_test保持一致
        if all_logits:
            placeholder_logit = all_logits[0]
            # 注意: 此处假设只需要返回有效的形状，而不关心内容
            merged_logits = placeholder_logit
        else:
            # 极端情况下创建一个空的占位符
            merged_logits = torch.zeros((1, 1, 256), device='cpu')
            
        logger.info(f"平均BPP: {avg_bpp:.4f}")
        
        return img_tensor, merged_logits, torch.tensor(avg_bpp, device=self.device)
        
    def model_forward(self, batch_patches):
        """GPT2+LoRA特定的模型前向传播方法"""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            current_batch_size = batch_patches.size(0)
            
            # 处理批次
            batch_pixels = batch_patches.long().clamp(0, 255).reshape(current_batch_size, -1)
            
            # 为自回归预测准备输入和标签
            inputs = batch_pixels[:, :-1]
            labels = batch_pixels[:, 1:]
            
            # 通过模型前向传播
            outputs = self.model(input_ids=inputs)
            hidden_states = outputs.last_hidden_state
            
            # 通过prob层获取预测
            batch_logits = self.base_model.prob(hidden_states)
            
            # 计算交叉熵损失
            loss = torch.nn.functional.cross_entropy(
                batch_logits.reshape(-1, 256),
                labels.reshape(-1)
            )
            
            # 计算BPP
            batch_bpp = loss.item() / np.log(2)
            
            return batch_logits, torch.tensor(batch_bpp, device=self.device)

    def visualize_results(self, original_img, reconstructed_img, metrics, idx):
        """与distillation_test保持一致的可视化函数，跳过实际可视化"""
        logger.info("跳过图像可视化 (仅需计算BPP)")
        # 可选：如果需要可视化，使用下面的代码
        if self.save_images:
            import matplotlib.pyplot as plt
            
            # 裁剪回原始尺寸
            h, w = self.original_h, self.original_w
            original_img = original_img[:, :, :h, :w]
            reconstructed_img = reconstructed_img[:, :, :h, :w]

            # 转换为可视化格式
            original = original_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            reconstructed = reconstructed_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            # 保存到LoRA结果目录
            original_path = f"{self.results_dir}/original_{idx}.png"
            reconstructed_path = f"{self.results_dir}/reconstructed_{idx}.png"
            plt.imsave(original_path, original)
            plt.imsave(reconstructed_path, reconstructed)
            
            logger.info(f"原始图像已保存至 {original_path}")
            logger.info(f"重建图像已保存至 {reconstructed_path}")
    
    def run(self, image_indices=None):
        """修改为与distillation_test一致的结果保存逻辑"""
        results = super().run(image_indices)
        
        # 如果有结果，保存到CSV
        if results:
            try:
                import csv
                import os
                
                # 获取全局 args 变量
                global args
                if 'args' in globals() and hasattr(args, 'dataset_name'):
                    dataset_name = args.dataset_name
                # 备用方法 (如果全局变量不可用)
                elif hasattr(self.dataset, 'image_files') and self.dataset.image_files:
                    dataset_name = os.path.basename(os.path.dirname(self.dataset.image_files[0]))
                else:
                    dataset_name = "dataset"
                
                # 创建输出目录
                os.makedirs(self.results_dir, exist_ok=True)
                
                # 使用与distillation_test一致的文件命名格式
                csv_path = f"{self.results_dir}/lora_{dataset_name}_metrics.csv"
                
                # 写入 CSV 文件
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Image Index", "PSNR", "Model BPP", "Actual BPP", "Compression Ratio"])
                    
                    for r in results:
                        writer.writerow([
                            r["image_idx"],
                            r["metrics"]["psnr"],
                            r["metrics"]["model_bpp"],
                            r["metrics"]["actual_bpp"] if "actual_bpp" in r["metrics"] and r["metrics"]["actual_bpp"] else "N/A",
                            r["metrics"]["compression_ratio"]
                        ])
                    logger.info(f"指标已保存到 {csv_path}")
            except Exception as e:
                logger.error(f"保存指标到CSV时出错: {e}")
        
        return results


if __name__ == "__main__":
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='测试LoRA微调的GPT2图像压缩')
        parser.add_argument('--model', type=str, default="/remote-home/wufeiyang/3_distill/4_distill_140_epoch/best_ce_model/model.pth", 
                            help='蒸馏模型路径')
        parser.add_argument('--lora_path', type=str, default="/remote-home/wufeiyang/4_dlstill_lora/4_distilled_lora/best_model", 
                            help='LoRA适配器权重路径')
        parser.add_argument('--gpt2_model_dir', type=str, default="/remote-home/wufeiyang/gpt2_model", 
                            help='GPT2模型目录路径')
        parser.add_argument('--dataset', type=str, default='/remote-home/wufeiyang/dataset/kodak', 
                            help='数据集路径')
        parser.add_argument('--skip_ac', action='store_true', default=True,
                            help='跳过算术编码，仅计算理论BPP')
        parser.add_argument('--images', type=str, default=None, 
                            help='要处理的图像索引，逗号分隔 (例如 "0,5,10")')
        parser.add_argument('--dataset_name', type=str, choices=['kodak', 'div2k', 'clic_mobile', 'clic_professional'], 
                            default='kodak', help='使用预定义数据集名称')
        parser.add_argument('--no_save_images', action='store_true', default=True,
                            help='不保存原始和重构图像')
        # 移除 keep_original_size 参数，始终使用 True
        parser.add_argument('--batch_size', type=int, default=16,
                            help='处理patch的批大小 (越大越快，但占用更多内存)')
        parser.add_argument('--single_gpu', action='store_true',
                            help='仅使用单个GPU进行推理 (用于调试或GPU内存问题)')
        parser.add_argument('--dataset_type', type=str, choices=['huggingface', 'directory'], default='directory',
                            help='数据集类型: huggingface 或 图像目录')
        
        args = parser.parse_args()
        
        # 定义数据集路径和类型
        dataset_paths = {
            'kodak': '/remote-home/wufeiyang/dataset/kodak_dataset/test',
            'div2k': '/remote-home/wufeiyang/dataset/DIV2K_valid_HR/DIV2K_valid_HR',
            'clic_mobile': '/remote-home/wufeiyang/dataset/clic_dataset/mobile_valid/valid',
            'clic_professional': '/remote-home/wufeiyang/dataset/clic_dataset/professional_valid/valid'
        }

        dataset_types = {
            'kodak': 'huggingface',  # Kodak是HuggingFace格式
            'div2k': 'directory',    # DIV2K是普通图片目录
            'clic_mobile': 'directory',
            'clic_professional': 'directory'
        }

        # 根据用户选择设置数据集路径
        dataset_path = dataset_paths.get(args.dataset_name, args.dataset)

        # 如果用户选择了预定义数据集，自动设置正确的数据集类型
        dataset_type = args.dataset_type
        if args.dataset_name in dataset_types:
            dataset_type = dataset_types[args.dataset_name]
            logger.info(f"自动使用'{dataset_type}'格式加载{args.dataset_name}数据集")

        # 根据数据集类型加载数据
        if dataset_type == 'huggingface':
            logger.info(f"从{dataset_path}加载HuggingFace数据集...")
            from datasets import load_from_disk
            dataset = load_from_disk(dataset_path)
        else:
            logger.info(f"从目录加载图像: {dataset_path}...")
            # 获取所有PNG和JPG图像
            image_files = sorted(glob.glob(f"{dataset_path}/*.png") + 
                                glob.glob(f"{dataset_path}/*.jpg") + 
                                glob.glob(f"{dataset_path}/*.jpeg"))
            
            if not image_files:
                raise ValueError(f"在 {dataset_path} 中未找到图像文件")
                
            logger.info(f"找到 {len(image_files)} 个图像")
            
            # 使用我们自己定义的DirectoryImageDataset
            dataset = DirectoryImageDataset(image_files)
        
        # 解析图像索引（如果提供）
        image_indices = None
        if args.images:
            image_indices = [int(idx) for idx in args.images.split(',')]
            logger.info(f"将处理特定图像: {image_indices}")
        
        # 运行测试 - 总是保持原始尺寸
        tester = GPT2LoRATester(
            model_path=args.model,
            dataset=dataset,
            gpt2_model_dir=args.gpt2_model_dir,
            skip_ac=args.skip_ac,
            lora_path=args.lora_path,
            save_images=not args.no_save_images,
            keep_original_size=True,  # 始终为True，与distillation_test保持一致 
            batch_size=args.batch_size,
            single_gpu=args.single_gpu
        )
        results = tester.run(image_indices=image_indices)
    
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
