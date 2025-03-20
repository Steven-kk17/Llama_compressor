import os
import torch
import torch.nn as nn
import argparse
from test import ModelTester
import math

# 创建专门存放蒸馏结果的目录
os.makedirs("./distilled_results", exist_ok=True)

class DistilledModelTester(ModelTester):
    def __init__(self, model_path="best_model/model.pth", dataset=None, 
                 skip_ac=False, save_images=True, keep_original_size=False, 
                 batch_size=16, single_gpu=False):
        # 存储批处理大小和GPU设置
        self.batch_size = batch_size
        self.single_gpu = single_gpu
        
        # 调用父类初始化
        super().__init__(model_path, dataset, None, skip_ac, save_images, keep_original_size)
        
    def _init_model(self):
        """初始化蒸馏后的GPT2模型并加载权重"""
        print(f"Loading distilled model from {self.model_path}")
        
        # 导入GPT2模型
        from gpt2 import GPT2PixelAR
        
        # 创建GPT2模型
        self.model = GPT2PixelAR()
        
        # 加载训练好的权重
        checkpoint = torch.load(self.model_path, map_location="cpu")
        
        # 处理checkpoint格式
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
            
        # 处理可能的module前缀
        if all(k.startswith("module.") for k in checkpoint.keys()):
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}
            
        # 加载模型权重
        self.model.load_state_dict(checkpoint, strict=False)
        
        # 多GPU支持
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1 and not self.single_gpu:
            print(f"Using {num_gpus} GPUs for inference")
            self.model = nn.DataParallel(self.model)
        else:
            if num_gpus > 1:
                print(f"Single GPU mode enforced (ignoring {num_gpus-1} additional GPUs)")
            else:
                print(f"Using single GPU for inference")
                
        self.model = self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        print(f"Model ready for inference on {self.device}")

    def process_dataset_image(self, img):
        """处理单张图像，切分为patches并限制最大区域大小"""
        # 应用变换
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        b, c, h, w = img_tensor.shape
        
        # 保存原始尺寸信息
        self.original_h, self.original_w = h, w
        
        # 如果保持原始尺寸，计算并应用填充
        if self.keep_original_size:
            # 计算需要的填充量
            pad_h = 0 if h % self.patch_size == 0 else self.patch_size - (h % self.patch_size)
            pad_w = 0 if w % self.patch_size == 0 else self.patch_size - (w % self.patch_size)
            
            if pad_h > 0 or pad_w > 0:
                print(f"Padding image from {h}x{w} to {h+pad_h}x{w+pad_w} to be divisible by patch size {self.patch_size}")
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
        else:
            # 非原始尺寸模式，所有像素都被视为"原始的"
            self.pixel_mask = torch.ones_like(img_tensor, dtype=torch.bool)
        
        # 更新尺寸（可能已经填充）
        b, c, h, w = img_tensor.shape
        
        # 确保可以被patch_size整除
        assert h % self.patch_size == 0 and w % self.patch_size == 0, f"Image dimensions {h}x{w} not divisible by patch size {self.patch_size}"
        
        # 计算需要多少个patch
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        total_patches = num_patches_h * num_patches_w
        
        # 设置最大区域大小 (以patch为单位)，不超过16x16个patch
        max_region_size = 16  # patches
        
        print(f"Processing image: {b}x{c}x{h}x{w}, split into {num_patches_h}x{num_patches_w} patches")
        if self.keep_original_size:
            print(f"Original dimensions: {self.original_h}x{self.original_w}")
        
        # 检查是否需要分块处理
        if num_patches_h > max_region_size or num_patches_w > max_region_size:
            print(f"Large image detected, processing in regions of at most {max_region_size}x{max_region_size} patches")
            
            # 分区域处理
            all_logits = []
            total_bpp = 0.0
            
            # 计算需要的区域数量
            num_regions_h = math.ceil(num_patches_h / max_region_size)
            num_regions_w = math.ceil(num_patches_w / max_region_size)
            total_regions = num_regions_h * num_regions_w
            
            print(f"Image will be processed in {total_regions} regions")
            
            # 逐区域处理
            for region_i in range(num_regions_h):
                for region_j in range(num_regions_w):
                    # 计算当前区域的patch范围
                    start_h = region_i * max_region_size
                    start_w = region_j * max_region_size
                    end_h = min(start_h + max_region_size, num_patches_h)
                    end_w = min(start_w + max_region_size, num_patches_w)
                    
                    region_patches_h = end_h - start_h
                    region_patches_w = end_w - start_w
                    
                    print(f"Processing region ({region_i},{region_j}): {region_patches_h}x{region_patches_w} patches")
                    
                    # 提取当前区域
                    region_img = img_tensor[:, :, 
                                    start_h*self.patch_size:end_h*self.patch_size,
                                    start_w*self.patch_size:end_w*self.patch_size]
                    
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
                    
                    # 区域处理
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            # 批量处理patches
                            for i in range(0, len(patches), self.batch_size):
                                batch_patches = torch.cat(patches[i:min(i+self.batch_size, len(patches))], dim=0)
                                current_batch_size = batch_patches.size(0)  # 获取实际的batch大小
                                
                                # 处理批次
                                batch_logits, batch_bpps = self.model(batch_patches, None)

                                # 更健壮的BPP处理
                                if isinstance(batch_bpps, (float, int)):
                                    region_bpp += batch_bpps * current_batch_size
                                elif isinstance(batch_bpps, torch.Tensor) and batch_bpps.numel() == 1:
                                    region_bpp += batch_bpps.item() * current_batch_size
                                elif isinstance(batch_bpps, torch.Tensor):
                                    if len(batch_bpps) == current_batch_size:
                                        # One BPP value per image - sum all values
                                        region_bpp += batch_bpps.sum().item()
                                    elif batch_bpps.dim() == 0:
                                        # Single scalar tensor
                                        region_bpp += batch_bpps.item() * current_batch_size
                                    elif len(batch_bpps) == torch.cuda.device_count():
                                        # Each GPU returned an average BPP value for its portion of the batch
                                        # We need to properly weight these values based on samples per GPU
                                        gpu_count = torch.cuda.device_count()
                                        
                                        # Calculate how many samples were on each GPU
                                        base_samples = current_batch_size // gpu_count
                                        extra = current_batch_size % gpu_count
                                        samples_per_gpu = [base_samples + (1 if i < extra else 0) for i in range(gpu_count)]
                                        
                                        # Weight the BPP by samples processed
                                        weighted_bpp = sum(bpp.item() * count for bpp, count in zip(batch_bpps, samples_per_gpu))
                                        region_bpp += weighted_bpp
                                    else:
                                        # Unknown format - use average and show warning
                                        print(f"Warning: Unexpected BPP tensor shape {batch_bpps.shape}")
                                        region_bpp += batch_bpps.mean().item() * current_batch_size
                                
                                # 收集结果
                                region_logits.extend(batch_logits.chunk(current_batch_size, dim=0))

                    
                    # 添加当前区域的结果
                    all_logits.extend(region_logits)
                    total_bpp += region_bpp
                    
                    # 清理内存
                    torch.cuda.empty_cache()
        else:
            # 图像足够小，可以按原来的方式处理
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
            
            # 小图像处理
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # 批量处理patches
                    for i in range(0, len(patches), self.batch_size):
                        batch_patches = torch.cat(patches[i:min(i+self.batch_size, len(patches))], dim=0)
                        current_batch_size = batch_patches.size(0)  # 获取实际的batch大小
                        
                        # 处理批次
                        batch_logits, batch_bpps = self.model(batch_patches, None)
                        
                        # 更健壮的BPP处理
                        if isinstance(batch_bpps, (float, int)):
                            # 如果是单个值，应用到所有patches
                            total_bpp += batch_bpps * current_batch_size
                        elif isinstance(batch_bpps, torch.Tensor) and batch_bpps.numel() == 1:
                            # 单元素张量
                            total_bpp += batch_bpps.item() * current_batch_size
                        elif isinstance(batch_bpps, torch.Tensor):
                            # 如果是张量，检查长度是否匹配
                            if len(batch_bpps) == current_batch_size:
                                total_bpp += batch_bpps.sum().item()
                            elif len(batch_bpps) == torch.cuda.device_count():
                                # Each GPU returned an average BPP value for its portion of the batch
                                gpu_count = torch.cuda.device_count()
                                
                                # Calculate how many samples were on each GPU
                                base_samples = current_batch_size // gpu_count
                                extra = current_batch_size % gpu_count
                                samples_per_gpu = [base_samples + (1 if i < extra else 0) for i in range(gpu_count)]
                                
                                # Weight the BPP by samples processed
                                weighted_bpp = sum(bpp.item() * count for bpp, count in zip(batch_bpps, samples_per_gpu))
                                total_bpp += weighted_bpp
                            else:
                                # 其他长度不匹配情况
                                total_bpp += batch_bpps[0].item() * current_batch_size
                                print(f"Warning: BPP tensor length ({len(batch_bpps)}) doesn't match batch size ({current_batch_size})")
                        else:
                            # 其他情况，使用batch大小乘以BPP
                            print(f"Warning: Unexpected BPP type: {type(batch_bpps)}")
                            total_bpp += float(batch_bpps) * current_batch_size
                        
                        # 收集结果
                        all_logits.extend(batch_logits.chunk(current_batch_size, dim=0))
        
        # 计算平均BPP
        avg_bpp = total_bpp / total_patches
        
        # 合并所有logits
        merged_logits = torch.cat(all_logits, dim=1)
        print(f"Merged logits shape: {merged_logits.shape}, Average BPP: {avg_bpp:.4f}")
        
        return img_tensor, merged_logits, torch.tensor(avg_bpp)
        
    def visualize_results(self, original_img, reconstructed_img, metrics, idx):
        """修改输出路径以存储蒸馏结果"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 如果保持原始尺寸，裁剪回原始尺寸
        if self.keep_original_size:
            h, w = self.original_h, self.original_w
            original_img = original_img[:, :, :h, :w]
            reconstructed_img = reconstructed_img[:, :, :h, :w]
        
        # 转换张量为图像
        original = original_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        reconstructed = reconstructed_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # 打印调试信息
        print(f"Original image shape: {original.shape}, range: [{original.min()} - {original.max()}]")
        print(f"Reconstructed image shape: {reconstructed.shape}, range: [{reconstructed.min()} - {reconstructed.max()}]")
        
        if self.save_images:  # 仅在启用时保存
            print(f"Visualizing results - Original range: [{original.min()}-{original.max()}], " +
                f"Reconstructed range: [{reconstructed.min()}-{reconstructed.max()}]")
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original)
            plt.title("Original")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed)
            plt.title(f"Distilled Reconstructed (PSNR: {metrics['psnr']:.2f} dB)")
            plt.axis('off')
            
            # 保存到蒸馏结果目录
            original_path = f"./distilled_results/original_{idx}.png"
            reconstructed_path = f"./distilled_results/reconstructed_{idx}.png"
            plt.imsave(original_path, original)
            plt.imsave(reconstructed_path, reconstructed)
            plt.close()
            
            print(f"Original image saved to {original_path}")
            print(f"Reconstructed image saved to {reconstructed_path}")
        
    def run(self, image_indices=None):
        """修改以保存结果到蒸馏特定的CSV文件"""
        results = super().run(image_indices)
        
        # 如果有结果，保存到蒸馏特定CSV
        if results:
            import csv
            with open("./distilled_results/distilled_metrics.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Image Index", "PSNR", "Model BPP", "Actual BPP", "Compression Ratio"])
                for r in results:
                    writer.writerow([
                        r["image_idx"],
                        r["metrics"]["psnr"],
                        r["metrics"]["model_bpp"],
                        r["metrics"]["actual_bpp"],
                        r["metrics"]["compression_ratio"]
                    ])
                print("Metrics saved to ./distilled_results/distilled_metrics.csv")
                
        return results


if __name__ == "__main__":
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='Test distilled GPT2 image compression model')
        # parser.add_argument('--model', type=str, default="./distilled_model/best_model/model.pth",
        #                 help='Path to the distilled model weights')
        parser.add_argument('--model', type=str, default="/remote-home/wufeiyang/distilled_model/best_model/model.pth",
                        help='Path to the distilled model weights')
        parser.add_argument('--dataset', type=str, default='/remote-home/wufeiyang/dataset/kodak_dataset/test', 
                        help='Path to the dataset')
        parser.add_argument('--skip_ac', action='store_true', default=True,
                        help='Skip arithmetic coding and just calculate theoretical BPP')
        parser.add_argument('--images', type=str, default=None, 
                        help='Comma-separated list of image indices to process (e.g., "0,5,10")')
        parser.add_argument('--dataset_type', type=str, choices=['huggingface', 'directory'], default='directory',
                        help='Type of dataset: huggingface or directory of images')
        parser.add_argument('--dataset_name', type=str, choices=['kodak', 'div2k', 'clic_mobile', 'clic_professional'], 
                        default='kodak', help='Name of predefined dataset to use')
        parser.add_argument('--no_save_images', action='store_true', default=True,
                        help='Do not save original and reconstructed images')
        parser.add_argument('--keep_original_size', action='store_true',
                        help='Keep original image dimensions (with padding to be divisible by patch size)')
        parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing patches (higher = faster, but more memory)')
        parser.add_argument('--single_gpu', action='store_true',
                        help='Use only a single GPU for inference (useful for debugging or GPU memory issues)')
        
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
        if args.dataset_name in dataset_types and args.dataset_type == 'directory':
            dataset_type = dataset_types[args.dataset_name]
            print(f"Automatically using '{dataset_type}' format for {args.dataset_name} dataset")

        # 根据数据集类型加载数据
        if dataset_type == 'huggingface':
            print(f"Loading HuggingFace dataset from {dataset_path}...")
            from datasets import load_from_disk
            dataset = load_from_disk(dataset_path)
        else:
            print(f"Loading images from directory: {dataset_path}...")
            # 创建简单的图像文件夹数据集
            from PIL import Image
            import glob
            
            # 获取所有PNG和JPG图像
            image_files = sorted(glob.glob(f"{dataset_path}/*.png") + 
                                glob.glob(f"{dataset_path}/*.jpg") + 
                                glob.glob(f"{dataset_path}/*.jpeg"))
            
            if not image_files:
                raise ValueError(f"No image files found in {dataset_path}")
                
            print(f"Found {len(image_files)} images")
            
            # 创建简单的类似数据集的字典结构
            class DirectoryImageDataset:
                def __init__(self, image_files):
                    self.image_files = image_files
                    
                def __len__(self):
                    return len(self.image_files)
                    
                def __getitem__(self, idx):
                    try:
                        img = Image.open(self.image_files[idx]).convert('RGB')
                        return {"image": img, "file": self.image_files[idx]}
                    except Exception as e:
                        print(f"Error loading image {self.image_files[idx]}: {e}")
                        # 返回一个小的黑色图像作为替代
                        return {"image": Image.new('RGB', (256, 256), color='black'), 
                                "file": self.image_files[idx]}
            
            dataset = DirectoryImageDataset(image_files)
        
        # 解析图像索引（如果提供）
        image_indices = None
        if args.images:
            image_indices = [int(idx) for idx in args.images.split(',')]
            print(f"将处理特定图像: {image_indices}")
        
        # 使用蒸馏模型运行测试
        tester = DistilledModelTester(
            model_path=args.model,
            dataset=dataset,
            skip_ac=args.skip_ac,
            save_images=not args.no_save_images,
            keep_original_size=args.keep_original_size,
            batch_size=args.batch_size,
            single_gpu=args.single_gpu
        )
        results = tester.run(image_indices=image_indices)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
