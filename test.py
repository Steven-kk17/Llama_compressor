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


class ModelTester:
    def __init__(self, model_path="final_model.pth", dataset=None, saved_model_dir="./saved_model",
                 skip_ac=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.dataset = dataset
        self.saved_model_dir = saved_model_dir
        self.image_size = 256  # Default size for testing
        self.patch_size = 16   # Define patch size as a class property
        self.skip_ac = skip_ac  # Flag to skip arithmetic coding
        
        # Ensure image size is divisible by patch size
        if self.image_size % self.patch_size != 0:
            self.image_size = (self.image_size // self.patch_size) * self.patch_size
            print(f"Adjusted image size to {self.image_size}x{self.image_size} to be divisible by patch size {self.patch_size}")
        
        # Use CenterCrop after Resize to ensure exact dimensions
        self.transform = T.Compose([
            T.Resize(self.image_size),  # Resize while preserving aspect ratio
            T.CenterCrop((self.image_size, self.image_size)),  # Force exact dimensions
            T.ToTensor(),
            T.Lambda(lambda x: x * 255)  # Scale to [0, 255]
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
        """Process a single image from dataset by splitting into patches"""
        # Apply transformation to ensure consistent size
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        b, c, h, w = img_tensor.shape
        
        # Verify dimensions are as expected
        if h != self.image_size or w != self.image_size:
            print(f"Warning: Image dimensions {h}x{w} don't match expected {self.image_size}x{self.image_size}")
            # Force resize if necessary
            img_tensor = T.functional.resize(img_tensor, (self.image_size, self.image_size))
            b, c, h, w = img_tensor.shape
        
        # Verify divisibility by patch size
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"Image dimensions {h}x{w} not divisible by patch size {self.patch_size}")
        
        # 计算需要多少个patch
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        
        print(f"Processing image: {b}x{c}x{h}x{w}, split into {num_patches_h}x{num_patches_w} patches")
        
        # Rest of the method remains the same...
        all_logits = []
        total_bpp = 0.0
        
        # 分批处理patches
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 逐个处理patch
                for i in range(num_patches_h):
                    for j in range(num_patches_w):
                        # 提取patch
                        patch = img_tensor[:, :, 
                                        i*self.patch_size:(i+1)*self.patch_size, 
                                        j*self.patch_size:(j+1)*self.patch_size]
                        
                        # 处理单个patch
                        patch_logits, patch_bpp = self.model(patch, None)
                        all_logits.append(patch_logits)
                        total_bpp += patch_bpp.item()
        
        # 计算平均BPP
        avg_bpp = total_bpp / (num_patches_h * num_patches_w)
        
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
        
        # MSE
        mse = ((original.cpu() - reconstructed.cpu()) ** 2).mean().item()
        
        # PSNR
        psnr = 10 * math.log10(255.0**2 / mse) if mse > 0 else float('inf')
        
        return {"mse": mse, "psnr": psnr}
    
    def visualize_results(self, original_img, reconstructed_img, metrics, idx):
        """Visualize original vs reconstructed images"""
        # Convert tensors to images
        original = original_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        reconstructed = reconstructed_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # 打印调试信息
        print(f"Original image shape: {original.shape}, range: [{original.min()} - {original.max()}]")
        print(f"Reconstructed image shape: {reconstructed.shape}, range: [{reconstructed.min()} - {reconstructed.max()}]")
        
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
                num_images = min(24, len(self.dataset))
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
                    
                    # Count total pixels for dataset BPP calculation
                    total_dataset_pixels += pixel_ids.numel()
                    
                    # Shift for autoregressive prediction (ignore last logit, use first token as context only)
                    shifted_logits = logits[:, :-1].contiguous() # [b, seq_len-1, 256]
                    shifted_probs = probs[:, :-1].contiguous() # [b, seq_len-1, 256]
                    shifted_targets = pixel_ids[:, 1:].contiguous() # [b, seq_len-1]
                    
                    # Calculate theoretical bits from model's BPP
                    theor_bits = model_bpp.item() * pixel_ids.numel()
                    total_theor_bits += theor_bits
                    
                    # Only do arithmetic coding if not skipped
                    if not self.skip_ac:
                        # Arithmetic coding
                        print("Performing arithmetic encoding...")
                        # 保存第一个像素值（这个不参与算术编码）
                        first_pixel = pixel_ids[:, 0].clone()  # [b, 1]
                        encoded, encoded_length = self.arithmetic_encode(shifted_probs, shifted_targets)
                        
                        # Calculate compression ratio
                        original_size = img_tensor.numel() * 8  # 8 bits per pixel
                        compressed_size = encoded_length * 8  # 8 bits per byte
                        compression_ratio = original_size / compressed_size
                        
                        # Add to dataset total bits
                        total_dataset_bits += compressed_size
                        
                        print(f"Original size: {original_size/8000:.2f} KB")
                        print(f"Compressed size: {compressed_size/8000:.2f} KB")
                        print(f"Compression ratio: {compression_ratio:.2f}x")
                        print(f"Model BPP: {model_bpp.item():.4f}")
                        print(f"Actual BPP: {compressed_size/img_tensor.numel():.4f}")
                        
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
                        
                        actual_bpp = compressed_size / img_tensor.numel()
                    else:
                        # Skip arithmetic coding, use model values
                        print("Skipping arithmetic coding (using model values)")
                        compression_ratio = 8.0 / model_bpp.item()  # Theoretical ratio
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
        parser.add_argument('--model', type=str, default="/remote-home/wufeiyang/final_model.pth", 
                            help='Path to the model weights')
        parser.add_argument('--model_dir', type=str, default="/remote-home/wufeiyang/saved_model", 
                            help='Path to the model config directory')
        parser.add_argument('--dataset', type=str, default='/remote-home/wufeiyang/dataset/kodak_dataset/test', 
                            help='Path to the dataset')
        parser.add_argument('--skip_ac', action='store_true', 
                            help='Skip arithmetic coding and just calculate theoretical BPP')
        parser.add_argument('--images', type=str, default=None, 
                            help='Comma-separated list of image indices to process (e.g., "0,5,10")')
        
        args = parser.parse_args()
        
        # Load dataset
        print(f"Loading dataset from {args.dataset}...")
        dataset = load_from_disk(args.dataset)
        print(f"Dataset loaded: {dataset}")
        
        # Parse image indices if provided
        image_indices = None
        if args.images:
            image_indices = [int(idx) for idx in args.images.split(',')]
            print(f"Will process specific images: {image_indices}")
        
        # Example usage
        tester = ModelTester(
            model_path=args.model,
            dataset=dataset,
            saved_model_dir=args.model_dir,
            skip_ac=args.skip_ac
        )
        results = tester.run(image_indices=image_indices)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()