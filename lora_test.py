import os
import torch
import torch.nn as nn
import argparse
from test import ModelTester
from peft import PeftModel
import math
# python lora_test.py --skip_ac --keep_original_size --batch_size 8 --no_save_images --single_gpu

# Create results directory specifically for LoRA results
os.makedirs("./lora_results", exist_ok=True)

class LoRAModelTester(ModelTester):
    def __init__(self, model_path="model_epoch_40.pth", dataset=None, saved_model_dir="./saved_model",
                 skip_ac=False, lora_path="/remote-home/wufeiyang/best_model", save_images=True, 
                 keep_original_size=False, batch_size=8, single_gpu=False):
        # Store LoRA path and batch size for optimization
        self.lora_path = lora_path
        self.batch_size = batch_size
        self.single_gpu = single_gpu
        
        # Call parent initializer with all parameters including keep_original_size
        super().__init__(model_path, dataset, saved_model_dir, skip_ac, save_images, keep_original_size)
        
    def _init_model(self):
        """Override to add LoRA adapters to the base model and merge weights for faster inference"""
        print(f"Loading base model from {self.model_path}")
        self.model = SafeLlamaPixelAR(model_path=self.saved_model_dir)
        
        # Load the trained weights
        checkpoint = torch.load(self.model_path, map_location="cpu")
        
        # Check if the checkpoint contains a module key (from DataParallel/DistributedDataParallel)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        
        # Handle potential prefix in state_dict keys
        if all(k.startswith("module.") for k in checkpoint.keys()):
            # Remove the 'module.' prefix
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}
        
        # Create a new state_dict to match model structure
        new_state_dict = {}
        
        # Rename keys to match current model structure
        for key, value in checkpoint.items():
            if key.startswith("llama.model."):
                # Replace "llama.model." with "llama."
                new_key = key.replace("llama.model.", "llama.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load the base model weights
        self.model.load_state_dict(new_state_dict, strict=False)
        print("Base model loaded successfully")
        
        # Extract the llama portion for LoRA adaptation
        print(f"Loading LoRA adapters from {self.lora_path}")
        llama_model = self.model.llama
        
        # Apply LoRA adapters to the model
        llama_model = PeftModel.from_pretrained(
            llama_model,
            self.lora_path,
            is_trainable=False  # We're just inferencing
        )
        
        # *** 关键优化 #1: 合并LoRA权重到基础模型 ***
        print("Merging LoRA weights into base model for faster inference...")
        llama_model = llama_model.merge_and_unload()
        print("LoRA weights merged successfully")
        
        # Replace original llama with LoRA-adapted version (now with merged weights)
        self.model.llama = llama_model
        
        # Check available GPUs and use DataParallel if multiple are available
        # 检查是否明确要求单GPU模式
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1 and not self.single_gpu:  # 添加single_gpu检查
            print(f"Using {num_gpus} GPUs for inference")
            self.model = nn.DataParallel(self.model)
        else:
            if num_gpus > 1:
                print(f"Single GPU mode enforced (ignoring {num_gpus-1} additional GPUs)")
            else:
                print(f"Using single GPU for inference")
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print(f"Model ready for inference on {self.device}")

    def process_dataset_image(self, img):
        """Process a single image from dataset by splitting into patches and limiting max region size"""
        # Apply transformation without resizing if keep_original_size is True
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        b, c, h, w = img_tensor.shape
        
        # 保存原始尺寸信息，用于后面的掩码和计算
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
                    
                    # 更新 process_dataset_image 方法中的 BPP 处理逻辑

                    # 大图像区域处理
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            # 批量处理patches
                            for i in range(0, len(patches), self.batch_size):
                                batch_patches = torch.cat(patches[i:min(i+self.batch_size, len(patches))], dim=0)
                                current_batch_size = batch_patches.size(0)  # 获取实际的batch大小
                                
                                # 处理批次
                                batch_logits, batch_bpps = self.model(batch_patches, None)
                                # print(f"BPP type: {type(batch_bpps)}, ", end="")
                                # if isinstance(batch_bpps, torch.Tensor):
                                #     print(f"shape: {batch_bpps.shape}, values: {batch_bpps}")
                                # else:
                                #     print(f"value: {batch_bpps}")

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
                                        
                                        # print(f"GPU distribution: {samples_per_gpu} samples, BPP values: {[b.item() for b in batch_bpps]}")
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
            
            # 修改小图像处理部分的BPP计算逻辑
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # 批量处理patches
                    for i in range(0, len(patches), self.batch_size):
                        batch_patches = torch.cat(patches[i:min(i+self.batch_size, len(patches))], dim=0)
                        current_batch_size = batch_patches.size(0)  # 获取实际的batch大小
                        
                        # 处理批次
                        batch_logits, batch_bpps = self.model(batch_patches, None)
                        # print(f"BPP type: {type(batch_bpps)}, ", end="")
                        # if isinstance(batch_bpps, torch.Tensor):
                        #     print(f"shape: {batch_bpps.shape}, values: {batch_bpps}")
                        # else:
                        #     print(f"value: {batch_bpps}")
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
                                
                                # print(f"GPU distribution: {samples_per_gpu} samples, BPP values: {[b.item() for b in batch_bpps]}")
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
        """Override to modify output paths for LoRA results and handle original size"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 如果保持原始尺寸，裁剪回原始尺寸
        if self.keep_original_size:
            h, w = self.original_h, self.original_w
            original_img = original_img[:, :, :h, :w]
            reconstructed_img = reconstructed_img[:, :, :h, :w]
        
        # Convert tensors to images
        original = original_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        reconstructed = reconstructed_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # Print debug info
        print(f"Original image shape: {original.shape}, range: [{original.min()} - {original.max()}]")
        print(f"Reconstructed image shape: {reconstructed.shape}, range: [{reconstructed.min()} - {reconstructed.max()}]")
        
        if self.save_images:  # Only save if enabled
            print(f"Visualizing results - Original range: [{original.min()}-{original.max()}], " +
                f"Reconstructed range: [{reconstructed.min()}-{reconstructed.max()}]")
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original)
            plt.title("Original")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed)
            plt.title(f"LoRA Reconstructed (PSNR: {metrics['psnr']:.2f} dB)")
            plt.axis('off')
            
            # Save to LoRA results directory
            original_path = f"./lora_results/original_{idx}.png"
            reconstructed_path = f"./lora_results/reconstructed_{idx}.png"
            plt.imsave(original_path, original)
            plt.imsave(reconstructed_path, reconstructed)
            plt.close()
            
            print(f"Original image saved to {original_path}")
            print(f"Reconstructed image saved to {reconstructed_path}")
        
    def run(self, image_indices=None):
        """Override to modify summary output"""
        results = super().run(image_indices)
        
        # If results exist, save to LoRA-specific CSV
        if results:
            import csv
            with open("./lora_results/lora_metrics.csv", "w", newline="") as f:
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
                print("Metrics saved to ./lora_results/lora_metrics.csv")
                
        return results


if __name__ == "__main__":
    try:
        # Import here to avoid circular imports
        from main import SafeLlamaPixelAR
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Test LoRA-adapted LLaMA image compression')
        parser.add_argument('--model', type=str, default="/remote-home/wufeiyang/model_epoch_40.pth", 
                            help='Path to the base model weights')
        parser.add_argument('--lora_path', type=str, default="/remote-home/wufeiyang/best_model", 
                            help='Path to the LoRA adapter weights')
        parser.add_argument('--model_dir', type=str, default="/remote-home/wufeiyang/saved_model", 
                            help='Path to the model config directory')
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
        parser.add_argument('--no_save_images', action='store_true',default=True,
                            help='Do not save original and reconstructed images')
        parser.add_argument('--keep_original_size', action='store_true',
                            help='Keep original image dimensions (with padding to be divisible by patch size)')
        parser.add_argument('--batch_size', type=int, default=8,
                            help='Batch size for processing patches (higher = faster, but more memory)')
        paras = parser.add_argument('--single_gpu', action='store_true',
                            help='Use only a single GPU for inference (useful for debugging or GPU memory issues)')
        
        args = parser.parse_args()
        
        # Define dataset paths and types
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
        
        # Parse image indices if provided
        image_indices = None
        if args.images:
            image_indices = [int(idx) for idx in args.images.split(',')]
            print(f"Will process specific images: {image_indices}")
        
        # Run test with LoRA model
        tester = LoRAModelTester(
            model_path=args.model,
            dataset=dataset,
            saved_model_dir=args.model_dir,
            skip_ac=args.skip_ac,
            lora_path=args.lora_path,
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

