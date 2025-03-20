import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import math
import sys
from main import SafeLlamaPixelAR
from datasets import load_from_disk
import yaecl

os.makedirs("./results", exist_ok=True)

# python test.py --keep_original_size --skip_ac --no_save_images --dataset_name kodak

class ModelTester:
    def __init__(self, model_path="model_epoch_40.pth", dataset=None, saved_model_dir="./saved_model",
                 skip_ac=True, save_images=False, keep_original_size=False, batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.dataset = dataset
        self.saved_model_dir = saved_model_dir
        self.image_size = 256  # Default size for testing
        self.patch_size = 16   # Define patch size as a class property
        self.skip_ac = skip_ac  # Flag to skip arithmetic coding
        self.save_images = save_images
        self.keep_original_size = keep_original_size
        self.batch_size = batch_size

        if self.keep_original_size:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Lambda(lambda x: x * 255)  # Scale to [0, 255]
            ])
        else:
            # 使用原有的调整大小转换
            self.transform = T.Compose([
                T.Resize(self.image_size),
                T.CenterCrop((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Lambda(lambda x: x * 255)
            ])
        
        # Initialize model and load weights
        self._init_model()
        
    def _init_model(self):
        print(f"Loading model from {self.model_path}")
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
        
        # 创建一个新的state_dict来匹配模型结构
        new_state_dict = {}
        
        # 重命名键以匹配当前模型结构
        for key, value in checkpoint.items():
            if key.startswith("llama.model."):
                # 将 "llama.model." 替换为 "llama."
                new_key = key.replace("llama.model.", "llama.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # 使用strict=False允许部分加载
        self.model.load_state_dict(new_state_dict, strict=False)
        
        # 检查可用GPU数量，如果有多个则使用DataParallel
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for inference")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print("Model loaded successfully")
    
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
                    
                    # 修复区域处理中的批处理部分
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
                                    region_bpp += batch_bpps * current_batch_size
                                elif isinstance(batch_bpps, torch.Tensor) and batch_bpps.numel() == 1:
                                    # 单元素张量
                                    region_bpp += batch_bpps.item() * current_batch_size
                                elif isinstance(batch_bpps, torch.Tensor):
                                    # 如果是张量，检查长度是否匹配
                                    if len(batch_bpps) == current_batch_size:
                                        region_bpp += batch_bpps.sum().item()
                                    elif len(batch_bpps) == torch.cuda.device_count():
                                        # Each GPU returned an average BPP value for its portion of the batch
                                        gpu_count = torch.cuda.device_count()
                                        
                                        # Calculate how many samples were on each GPU
                                        base_samples = current_batch_size // gpu_count
                                        extra = current_batch_size % gpu_count
                                        samples_per_gpu = [base_samples + (1 if i < extra else 0) for i in range(gpu_count)]
                                        
                                        # Weight the BPP by samples processed
                                        weighted_bpp = sum(bpp.item() * count for bpp, count in zip(batch_bpps, samples_per_gpu))
                                        region_bpp += weighted_bpp  # or total_bpp += weighted_bpp for small images
                                        
                                        # print(f"GPU distribution: {samples_per_gpu} samples, BPP values: {[b.item() for b in batch_bpps]}")
                                    else:
                                        # 长度不匹配，使用第一个值乘以batch大小
                                        region_bpp += batch_bpps[0].item() * current_batch_size
                                        print(f"Warning: BPP tensor length ({len(batch_bpps)}) doesn't match batch size ({current_batch_size})")
                                else:
                                    # 其他情况，使用batch大小乘以BPP
                                    print(f"Warning: Unexpected BPP type: {type(batch_bpps)}")
                                    region_bpp += float(batch_bpps) * current_batch_size
                                
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
                                total_bpp += weighted_bpp  # or total_bpp += weighted_bpp for small images
                                
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
        
    def get_probabilities(self, logits):
        """Convert logits to probabilities"""
        return torch.nn.functional.softmax(logits, dim=-1)
        
    def arithmetic_encode(self, probabilities, targets):
        """Use yaecl arithmetic coding to encode the image"""
        try:
            # 确保转换为float32，然后转为NumPy
            probs = probabilities.cpu().float().numpy()
            targets_np = targets.cpu().numpy().astype(np.int32)

            print(f"Encoding - Probabilities shape: {probs.shape}, Targets shape: {targets_np.shape}")
            
            # 准备CDF数组
            cdf_bits = 16
            cdf_max = 2**cdf_bits
            batch_size, seq_len, num_classes = probs.shape  # [B, seq_len - 1, 256]
            
            # 创建CDF数组
            cdf_array = np.zeros((batch_size, seq_len, num_classes+1), dtype=np.int32)
            
            # 填充CDF数组
            for b in range(batch_size):
                for s in range(seq_len):
                    # 确保概率和为1，防止浮点误差
                    probs_norm = probs[b, s] / np.sum(probs[b, s])
                    
                    # 计算累积概率
                    cumsum = np.cumsum(probs_norm)
                    
                    # 创建严格递增的CDF
                    cdf_array[b, s, 0] = 0  # CDF始终从0开始
                    for i in range(num_classes):
                        # 确保CDF严格递增，至少增加1
                        prev_val = cdf_array[b, s, i]
                        curr_val = int(cumsum[i] * cdf_max)
                        cdf_array[b, s, i+1] = max(prev_val + 1, curr_val)
                    
                    # 确保最后一个值正好是cdf_max
                    cdf_array[b, s, -1] = cdf_max

            # 创建yaecl编码器
            ac_enc = yaecl.ac_encoder_t()
            
            # 使用nxn编码模式
            ac_enc.encode_nxn(memoryview(targets_np.reshape(-1)), 
                            memoryview(cdf_array.reshape(-1, num_classes+1)), 
                            cdf_bits)
            ac_enc.flush()
            
            # 获取编码后的比特流
            encoded_data = ac_enc.bit_stream
            
            # 使用实际比特流大小
            actual_bytes = len(encoded_data)
            
            print(f"Encoding complete - actual bitstream size: {actual_bytes} bytes")
            return encoded_data, actual_bytes
        except Exception as e:
            print(f"Error in arithmetic encoding: {e}")
            # 使用理论BPP估计的压缩大小
            approx_bytes = int(targets_np.size * 5.5 / 8)
            print(f"Using fallback size estimate: ~{approx_bytes} bytes")
            return None, approx_bytes

    def arithmetic_decode(self, encoded_data, probabilities, length):
        """Decode the arithmetic coded data using yaecl"""
        try:
            print(f"Decoding...")
            if encoded_data is None:
                print("No encoded data available, skipping decoding")
                return np.zeros(length, dtype=np.int32)
                
            # 确保转换为float32，然后转为NumPy
            probs = probabilities.cpu().float().numpy()
            
            # 准备CDF数组
            cdf_bits = 16
            cdf_max = 2**cdf_bits
            batch_size, seq_len, num_classes = probs.shape
            
            # 创建CDF数组
            cdf_array = np.zeros((batch_size, seq_len, num_classes+1), dtype=np.int32)
            
            # 填充CDF数组 - 使用与编码相同的方法确保一致性
            for b in range(batch_size):
                for s in range(seq_len):
                    # 确保概率和为1，防止浮点误差
                    probs_norm = probs[b, s] / np.sum(probs[b, s])
                    
                    # 计算累积概率
                    cumsum = np.cumsum(probs_norm)
                    
                    # 创建严格递增的CDF
                    cdf_array[b, s, 0] = 0  # CDF始终从0开始
                    for i in range(num_classes):
                        # 确保CDF严格递增，至少增加1
                        prev_val = cdf_array[b, s, i]
                        curr_val = int(cumsum[i] * cdf_max)
                        cdf_array[b, s, i+1] = max(prev_val + 1, curr_val)
                    
                    # 确保最后一个值正好是cdf_max
                    cdf_array[b, s, -1] = cdf_max
            
            # 创建解码器
            ac_dec = yaecl.ac_decoder_t(encoded_data)
            
            # 创建输出数组
            decoded = np.zeros(length, dtype=np.int32)
            
            # 使用nxn解码模式
            ac_dec.decode_nxn(num_classes, memoryview(cdf_array.reshape(-1, num_classes+1)), cdf_bits, memoryview(decoded))
            
            print(f"Decoding complete")
            return decoded
        except Exception as e:
            print(f"Error in arithmetic decoding: {e}")
            # Return original targets if available (for testing)
            return np.zeros(length, dtype=np.int32)
    
    def calculate_metrics(self, original, reconstructed):
        """Calculate metrics between original and reconstructed images"""
        # 打印调试信息
        print(f"Original image shape: {original.shape}, range: [{original.min()} - {original.max()}]")
        print(f"Reconstructed image shape: {reconstructed.shape}, range: [{reconstructed.min()} - {reconstructed.max()}]")
        
        # 如果保持原始尺寸，只计算原始像素区域的指标
        if self.keep_original_size:
            # 使用掩码只选择原始像素
            mask = self.pixel_mask
            original_masked = original[mask].cpu()
            reconstructed_masked = reconstructed[mask].cpu()
            
            # MSE和PSNR只考虑原始像素
            mse = ((original_masked - reconstructed_masked) ** 2).mean().item()
        else:
            # 原来的计算方法
            mse = ((original.cpu() - reconstructed.cpu()) ** 2).mean().item()
        
        # PSNR
        psnr = 10 * math.log10(255.0**2 / mse) if mse > 0 else float('inf')
        
        return {"mse": mse, "psnr": psnr}
    
    def visualize_results(self, original_img, reconstructed_img, metrics, idx):
        """Visualize original vs reconstructed images"""
        if self.keep_original_size:
            h, w = self.original_h, self.original_w
            original_img = original_img[:, :, :h, :w]
            reconstructed_img = reconstructed_img[:, :, :h, :w]

        # Convert tensors to images
        original = original_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        reconstructed = reconstructed_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # 打印调试信息
        print(f"Original image shape: {original.shape}, range: [{original.min()} - {original.max()}]")
        print(f"Reconstructed image shape: {reconstructed.shape}, range: [{reconstructed.min()} - {reconstructed.max()}]")
        
        if self.save_images:
            print(f"Visualizing results - Original range: [{original.min()}-{original.max()}], " +
                f"Reconstructed range: [{reconstructed.min()}-{reconstructed.max()}]")
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original)
            plt.title("Original")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed)
            plt.title(f"Reconstructed (PSNR: {metrics['psnr']:.2f} dB)")
            plt.axis('off')
            
            # 保存到results目录
            original_path = f"./results/original_{idx}.png"
            reconstructed_path = f"./results/reconstructed_{idx}.png"
            plt.imsave(original_path, original)
            plt.imsave(reconstructed_path, reconstructed)
            plt.close()
            
            print(f"Original image saved to {original_path}")
            print(f"Reconstructed image saved to {reconstructed_path}")
    
    def run(self, image_indices=None):
        """
        Run the testing process
        
        Args:
            image_indices: List of specific image indices to process, or None to process all
        """
        results = []
        
        # Add variables to track dataset-wide statistics
        total_dataset_pixels = 0
        total_dataset_bits = 0
        total_theor_bits = 0
        
        # Use dataset images
        if self.dataset:
            # If specific indices provided, use those
            if image_indices is not None:
                indices = image_indices
                print(f"Processing {len(indices)} specific images: {indices}")
            else:
                # Otherwise use all images up to a limit
                num_images = len(self.dataset)
                indices = list(range(num_images))
                print(f"Processing all {num_images} images")
            
            for idx in tqdm(indices, desc="Processing images"):
                try:
                    print(f"\nProcessing image {idx+1}/{len(indices)} (index {idx})")
                    
                    # Get image from dataset
                    img = self.dataset[idx]["image"]
                    
                    # Process image
                    img_tensor, logits, model_bpp = self.process_dataset_image(img)
                    
                    # Get probabilities
                    probs = self.get_probabilities(logits)
                    
                    # Get shifted targets (original image pixels, shifted to predict next pixel)
                    pixel_ids = img_tensor.long().clamp(0, 255)
                    b, c, h, w = pixel_ids.shape
                    seq_len = c * h * w
                    pixel_ids = pixel_ids.view(b, seq_len)
                    
                    # 计算实际像素数量，考虑填充的情况
                    if self.keep_original_size:
                        # 计算原始像素数量（不包括填充部分）
                        original_pixels = self.original_h * self.original_w * c
                        print(f"Original pixel count: {original_pixels}, padded pixel count: {pixel_ids.numel()}")
                        # 只统计原始像素数量
                        total_dataset_pixels += original_pixels
                        
                        # 计算理论比特数，基于原始像素
                        theor_bits = model_bpp.item() * original_pixels
                    else:
                        # 原有逻辑，计算所有像素
                        total_dataset_pixels += pixel_ids.numel()
                        theor_bits = model_bpp.item() * pixel_ids.numel()
                    
                    total_theor_bits += theor_bits
                    
                    # Shift for autoregressive prediction (ignore last logit, use first token as context only)
                    shifted_logits = logits[:, :-1].contiguous() # [b, seq_len-1, 256]
                    shifted_probs = probs[:, :-1].contiguous() # [b, seq_len-1, 256]
                    shifted_targets = pixel_ids[:, 1:].contiguous() # [b, seq_len-1]
                    
                    # Only do arithmetic coding if not skipped
                    if not self.skip_ac:
                        # Arithmetic coding
                        print("Performing arithmetic encoding...")
                        # 保存第一个像素值（这个不参与算术编码）
                        first_pixel = pixel_ids[:, 0].clone()  # [b, 1]
                        encoded, encoded_length = self.arithmetic_encode(shifted_probs, shifted_targets)
                        
                        # 修改压缩比计算，考虑原始像素
                        if self.keep_original_size:
                            # 只计算原始图像区域的原始大小
                            original_size = original_pixels * 8  # 8 bits per pixel
                        else:
                            # 原来的计算
                            original_size = img_tensor.numel() * 8  # 8 bits per pixel
                        
                        compressed_size = encoded_length * 8  # 8 bits per byte
                        compression_ratio = original_size / compressed_size
                        
                        # Add to dataset total bits
                        total_dataset_bits += compressed_size
                        
                        # 修改BPP计算，基于原始像素
                        if self.keep_original_size:
                            actual_bpp = compressed_size / original_pixels
                        else:
                            actual_bpp = compressed_size / img_tensor.numel()
                        
                        print(f"Original size: {original_size/8000:.2f} KB")
                        print(f"Compressed size: {compressed_size/8000:.2f} KB")
                        print(f"Compression ratio: {compression_ratio:.2f}x")
                        print(f"Model BPP: {model_bpp.item():.4f}")
                        print(f"Actual BPP: {actual_bpp:.4f}")
                        
                        # Arithmetic decoding
                        print("Performing arithmetic decoding...")
                        decoded = self.arithmetic_decode(encoded, shifted_probs, shifted_targets.numel())
                        
                        # Check if decoded matches target
                        if encoded is not None:
                            decoded_tensor = torch.tensor(decoded, dtype=torch.long)
                            original_tensor = shifted_targets.cpu()
                            match_percentage = 100.0 * torch.sum(decoded_tensor == original_tensor).item() / original_tensor.numel()
                            print(f"Decoded data matches original: {match_percentage:.2f}%")
                        
                        # 将第一个像素与解码结果合并
                        full_pixels = torch.zeros(b, seq_len, device=self.device, dtype=torch.float32)
                        full_pixels[:, 0] = first_pixel
                        full_pixels[:, 1:] = torch.tensor(decoded, dtype=torch.float32).reshape(b, seq_len-1)
                    else:
                        # Skip arithmetic coding, use model values
                        print("Skipping arithmetic coding (using model values)")
                        
                        # 修改理论压缩比计算，考虑原始像素
                        if self.keep_original_size:
                            compression_ratio = 8.0 / model_bpp.item()  # 理论上的压缩比仍然是8比特除以BPP
                        else:
                            compression_ratio = 8.0 / model_bpp.item()
                        
                        actual_bpp = model_bpp.item()
                        # Just use original pixels for visualization
                        full_pixels = pixel_ids.float()

                    # 重建完整图像
                    reconstructed = full_pixels.reshape(b, c, h, w)
                    
                    # 打印调试信息
                    print(f"Reconstructed image shape: {reconstructed.shape}, range: [{reconstructed.min()} - {reconstructed.max()}]")
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(img_tensor, reconstructed)
                    metrics["model_bpp"] = model_bpp.item()
                    metrics["actual_bpp"] = actual_bpp
                    metrics["compression_ratio"] = compression_ratio
                    
                    print(f"PSNR: {metrics['psnr']:.2f} dB")
                    
                    # Visualize
                    self.visualize_results(img_tensor, reconstructed, metrics, idx)
                    
                    results.append({
                        "image_idx": idx,
                        "metrics": metrics
                    })
                except Exception as e:
                    print(f"Error processing image {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Always clean up memory
                    torch.cuda.empty_cache()
            
            # Calculate average metrics
            if results:
                avg_psnr = sum(r["metrics"]["psnr"] for r in results) / len(results)
                avg_model_bpp = sum(r["metrics"]["model_bpp"] for r in results) / len(results)
                avg_actual_bpp = sum(r["metrics"]["actual_bpp"] for r in results) / len(results)
                avg_ratio = sum(r["metrics"]["compression_ratio"] for r in results) / len(results)
                
                # Calculate dataset-wide BPP (both theoretical and actual)
                theor_dataset_bpp = total_theor_bits / total_dataset_pixels
                if total_dataset_bits > 0:
                    actual_dataset_bpp = total_dataset_bits / total_dataset_pixels
                else:
                    actual_dataset_bpp = theor_dataset_bpp
                
                print("\nSummary:")
                print(f"Average PSNR: {avg_psnr:.2f} dB")
                print(f"Average Model-Estimated BPP: {avg_model_bpp:.4f}")
                print(f"Average Actual BPP: {avg_actual_bpp:.4f}")
                print(f"Average Compression Ratio: {avg_ratio:.2f}x")
                print(f"Overall Theoretical Dataset BPP: {theor_dataset_bpp:.4f}")
                if not self.skip_ac:
                    print(f"Overall Actual Dataset BPP: {actual_dataset_bpp:.4f}")
            
        return results
        
        
if __name__ == "__main__":
    try:
        import argparse
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Test LLaMA image compression')
        parser.add_argument('--keep_original_size', action='store_true',
                    help='Keep original image dimensions (with padding to be divisible by patch size)')
        parser.add_argument('--model', type=str, default="/remote-home/wufeiyang/model_epoch_40.pth", 
                            help='Path to the model weights')
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
        parser.add_argument('--no_save_images', action='store_true', default=True,
                            help='Do not save original and reconstructed images')
        parser.add_argument('--batch_size', type=int, default=8,
                            help='Batch size for processing patches')


        args = parser.parse_args()
        
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
        
        tester = ModelTester(
            model_path=args.model,
            dataset=dataset,
            saved_model_dir=args.model_dir,
            skip_ac=args.skip_ac,
            save_images=not args.no_save_images,
            keep_original_size=args.keep_original_size,
            batch_size=args.batch_size
        )
        results = tester.run(image_indices=image_indices)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
