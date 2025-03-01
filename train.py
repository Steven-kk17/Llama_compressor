import os
import glob
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

from main import SafeLlamaPixelAR

# ---------- 数据集定义 ----------
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, size=128, patch_size=16):
        """
        root_dir: 存放原始图像的文件夹
        size:     将图像统一缩放为 size×size
        patch_size: 用于切分小块
        """
        super().__init__()
        self.root_dir = root_dir
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.size = size
        self.patch_size = patch_size
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),               # 转为 [0,1]
            T.Lambda(lambda x: x * 255)  # [0,255]
        ])

    def __len__(self):
        return len(self.paths)

    def patchify(self, img_tensor):
        """
        将 [3, size, size] 切分为若干 [3, patch_size, patch_size] 小块。
        返回形状: [num_patch, 3, patch_size, patch_size]
        """
        c, h, w = img_tensor.shape
        patches = []
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = img_tensor[:, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        return torch.stack(patches, dim=0)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)   # [3, size, size]
        patches = self.patchify(img_tensor)  # [num_patches, 3, patch_size, patch_size]
        return patches, img_tensor  # 同时返回原图（用于后续的可视化）

# ---------- 可视化函数 ----------
def unpatchify(patches, original_size=(64, 64), patch_size=16):
    """
    将 [num_patches, 3, patch_size, patch_size] 小块还原为 [3, H, W]。
    适用于能被 patch_size 整除的尺寸。
    """
    num_patches, c, _, _ = patches.shape
    h, w = original_size
    patches_per_row = w // patch_size
    rows = []
    for i in range(0, num_patches, patches_per_row):
        row = patches[i: i + patches_per_row]
        rows.append(torch.cat(list(row), dim=-1))
    return torch.cat(rows, dim=-2)

# ---------- 训练脚本 ----------
def train_model():
    # 参数设置
    root_dir = "/home/steven/code/vqllama/data/Flickr30K/images"  # 数据集文件夹
    batch_size = 1    # 每个样本为一张图（含多个 patch）
    epochs = 3
    lr = 1e-4
    size = 64      # 图像尺寸
    patch_size = 16

    # 构建数据集和 DataLoader
    dataset = ImageFolderDataset(root_dir, size=size, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载最新的模型，注意模型内部已冻结除 embedding 层（embed_tokens）外的所有参数
    model = SafeLlamaPixelAR(model_path="./saved_model").to(device).bfloat16()
    model.train()

    # （可选）打印待训练参数数量
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters count: {sum(p.numel() for p in trainable_params)}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # 训练循环
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0

        for batch in dataloader:
            # 每个 batch 返回 (patches, original_img)
            # patches: [batch_size, num_patches, 3, patch_size, patch_size]
            patches, _ = batch
            b, n, c, ph, pw = patches.shape
            # 将所有 patch 展平成 [b*n, 3, patch_size, patch_size]
            input_patches = patches.view(b * n, c, ph, pw).to(device, dtype=torch.bfloat16)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, ce_loss = model(input_patches, None)
            ce_loss.backward()
            optimizer.step()

            total_loss += ce_loss.item()
            steps += 1

        avg_loss = total_loss / steps
        print(f"[Epoch {epoch+1}/{epochs}] Avg CE loss (bpp): {avg_loss:.5f}")

    # 保存训练后的模型参数
    torch.save(model.state_dict(), "trained_llama_pixelar.pt")
    print("Training done. Model saved to trained_llama_pixelar.pt")

    # ---------- 可视化重建效果 ----------
    model.eval()
    with torch.no_grad():
        # 随机取一张图进行测试
        sample_patches, orig_img = dataset[0]  # sample_patches: [num_patches, 3, patch_size, patch_size]
        input_patches = sample_patches.to(device, dtype=torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(input_patches, None)
        # logits 的形状：[num_patches, seq_len, 256] 其中 seq_len = 3 * patch_size * patch_size
        # 取 argmax 得到预测的 token 序列
        pred_tokens = logits.argmax(dim=-1)  # [num_patches, 3*patch_size*patch_size]
        # 将 token 序列重构为 [3, patch_size, patch_size]
        pred_patches = pred_tokens.view(-1, 3, patch_size, patch_size).float().clamp(0, 255)
        reconstructed_img = unpatchify(pred_patches.cpu(), original_size=(size, size), patch_size=patch_size)

    # 可视化原图和重建图
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img.permute(1, 2, 0).byte().numpy())
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img.permute(1, 2, 0).byte().numpy())
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_model()