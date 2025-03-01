
# filepath: /home/steven/code/llama_compression/inference.py
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

from main import SafeLlamaPixelAR  # 假设 main.py 在同目录，可根据实际路径调整

def load_and_preprocess_image(img_path, size=256):
    """
    从指定路径加载图像，将其转为RGB并缩放为所需尺寸，
    返回一个 [3, size, size] 的张量，像素范围 [0,255]。
    """
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),                 # => [0, 1]
        T.Lambda(lambda x: x * 255)   # => [0, 255]
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)

def patchify(img_tensor, patch_size=16):
    """
    将 [3, H, W] 切分成若干 [3, patch_size, patch_size] 小块。
    输出: [num_patches, 3, patch_size, patch_size]
    """
    c, h, w = img_tensor.shape
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img_tensor[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return torch.stack(patches, dim=0)

def unpatchify(patches, original_size=(128,128), patch_size=16):
    """
    将 [num_patches, 3, patch_size, patch_size] 小块还原为 [3, H, W]。
    适用能被 patch_size 整除的尺寸。
    """
    num_patches, c, _, _ = patches.shape
    h, w = original_size
    patches_per_row = w // patch_size
    rows = []
    for i in range(0, num_patches, patches_per_row):
        row = patches[i : i + patches_per_row]
        rows.append(torch.cat(list(row), dim=-1))
    return torch.cat(rows, dim=-2)

if __name__ == "__main__":
    # 1) 读取并预处理图像
    img_dir = "/home/steven/code/vqllama/data/Flickr30K/images"
    img_file = "134206.jpg"   # 请替换为实际文件名
    img_path = os.path.join(img_dir, img_file)
    img_tensor = load_and_preprocess_image(img_path, size=128)  # [3,256,256]

    # 2) 切分为 16×16 小块，示例把每个小块当成一个输入批次
    patches_tensor = patchify(img_tensor, patch_size=16)        # [N,3,16,16]

    # 3) 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafeLlamaPixelAR("./saved_model").to(device).bfloat16()
    model.eval()

    # 4) 前向推理 (每个patch做一次逐像素自回归)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, bpp = model(patches_tensor.to(device), None)

    print("Patch logits shape:", logits.shape)  # 形如 [N, (3*16*16), 256]
    print("Avg BPP value:", bpp.item())

    # 5) 简易重建：对logits做 argmax，得到估计像素序列
    pred_patches = logits.argmax(dim=-1).view(-1, 3, 16, 16).float().clamp(0, 255)
    reconstructed = unpatchify(pred_patches.cpu(), (128, 128), patch_size=16)

    # 6) 可视化原图 vs. 简易重建
    plt.subplot(1, 2, 1)
    plt.imshow(img_tensor.permute(1, 2, 0).byte())
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.permute(1, 2, 0).byte())
    plt.title("Reconstructed")
    plt.axis("off")
    plt.show()
