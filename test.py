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

class ModelTester:
    def __init__(self, model_path="final_model.pth", dataset=None, saved_model_dir="./saved_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.dataset = dataset
        self.saved_model_dir = saved_model_dir
        self.image_size = 256  # Default size for testing
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
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
            
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print("Model loaded successfully")
    
    def process_dataset_image(self, img):
        """Process a single image from dataset and return the logits"""
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                               dtype=torch.bfloat16):
                logits, bpp = self.model(img_tensor, None)
        
        return img_tensor, logits, bpp
        
    def get_probabilities(self, logits):
        """Convert logits to probabilities"""
        return torch.nn.functional.softmax(logits, dim=-1)
    
    def arithmetic_encode(self, probabilities, targets):
        """Use arithmetic coding to encode the image"""
        # Convert to numpy
        probs_np = probabilities.cpu().numpy()
        targets_np = targets.cpu().numpy().astype(np.int32)
        
        # Encode each pixel
        encoded = yaecl.encode(targets_np.flatten(), probs_np.reshape(-1, 256))
        
        return encoded, len(encoded)
    
    def arithmetic_decode(self, encoded, probabilities, length):
        """Decode the arithmetic coded data"""
        # Convert to numpy
        probs_np = probabilities.cpu().numpy()
        
        # Decode
        decoded = yaecl.decode(encoded, probs_np.reshape(-1, 256), length)
        
        return decoded
    
    def calculate_metrics(self, original, reconstructed):
        """Calculate metrics between original and reconstructed images"""
        # MSE
        mse = ((original.cpu() - reconstructed.cpu()) ** 2).mean().item()
        
        # PSNR
        psnr = 10 * math.log10(255.0**2 / mse) if mse > 0 else float('inf')
        
        return {"mse": mse, "psnr": psnr}
    
    def visualize_results(self, original_img, reconstructed_img, metrics, idx):
        """Visualize original vs reconstructed images"""
        # Convert tensors to images
        original = original_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        reconstructed = reconstructed_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed)
        plt.title(f"Reconstructed (PSNR: {metrics['psnr']:.2f} dB)")
        plt.axis('off')
        
        plt.savefig(f"result_{idx}.png")
        plt.close()
    
    def run(self):
        results = []
        
        # Use dataset images
        if self.dataset:
            num_images = min(1, len(self.dataset))  # 限制为10张图像
            print(f"Using {num_images} images from dataset")
            
            for idx in tqdm(range(num_images), desc="Processing dataset images"):
                print(f"\nProcessing image {idx+1}/{num_images}")
                
                # Get image from dataset
                img = self.dataset[idx]["image"]
                
                # Process image
                img_tensor, logits, bpp = self.process_dataset_image(img)
                
                # Get probabilities
                probs = self.get_probabilities(logits)
                
                # Get shifted targets (original image pixels, shifted to predict next pixel)
                pixel_ids = img_tensor.long().clamp(0, 255)
                b, c, h, w = pixel_ids.shape
                seq_len = c * h * w
                pixel_ids = pixel_ids.view(b, seq_len)
                
                # Shift for autoregressive prediction (ignore last logit, use first token as context only)
                shifted_logits = logits[:, :-1].contiguous()
                shifted_probs = probs[:, :-1].contiguous()
                shifted_targets = pixel_ids[:, 1:].contiguous()
                
                # Arithmetic coding
                print("Performing arithmetic encoding...")
                encoded, encoded_length = self.arithmetic_encode(shifted_probs, shifted_targets)
                
                # Calculate compression ratio
                original_size = img_tensor.numel() * 8  # 8 bits per pixel
                compressed_size = len(encoded) * 8  # 8 bits per byte
                compression_ratio = original_size / compressed_size
                
                print(f"Original size: {original_size/8000:.2f} KB")
                print(f"Compressed size: {compressed_size/8000:.2f} KB")
                print(f"Compression ratio: {compression_ratio:.2f}x")
                print(f"BPP: {bpp.item():.4f}")
                
                # Arithmetic decoding
                print("Performing arithmetic decoding...")
                decoded = self.arithmetic_decode(encoded, shifted_probs, shifted_targets.numel())
                
                # Reshape decoded data back to image
                reconstructed = torch.tensor(decoded, dtype=torch.float32)
                reconstructed = reconstructed.reshape(b, c, h, w)
                
                # Calculate metrics
                metrics = self.calculate_metrics(img_tensor, reconstructed)
                metrics["bpp"] = bpp.item()
                metrics["compression_ratio"] = compression_ratio
                
                print(f"PSNR: {metrics['psnr']:.2f} dB")
                
                # Visualize
                self.visualize_results(img_tensor, reconstructed, metrics, idx)
                
                results.append({
                    "image_idx": idx,
                    "metrics": metrics
                })
            
            # Calculate average metrics
            avg_psnr = sum(r["metrics"]["psnr"] for r in results) / len(results)
            avg_bpp = sum(r["metrics"]["bpp"] for r in results) / len(results)
            avg_ratio = sum(r["metrics"]["compression_ratio"] for r in results) / len(results)
            
            print("\nSummary:")
            print(f"Average PSNR: {avg_psnr:.2f} dB")
            print(f"Average BPP: {avg_bpp:.4f}")
            print(f"Average Compression Ratio: {avg_ratio:.2f}x")
            
        return results
        
if __name__ == "__main__":
    try:
        # Load dataset
        print("Loading Kodak dataset...")
        dataset = load_from_disk('/remote-home/wufeiyang/dataset/kodak_dataset/test')
        print(f"Dataset loaded: {dataset}")
        
        # Create results directory if it doesn't exist
        os.makedirs("./results", exist_ok=True)
        
        # Example usage
        tester = ModelTester(
            model_path="/remote-home/wufeiyang/final_model.pth",
            dataset=dataset,  # Pass the dataset directly
            saved_model_dir="/remote-home/wufeiyang/saved_model"  # Path to base model config
        )
        results = tester.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()