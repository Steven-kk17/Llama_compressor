# filepath: /home/steven/code/llama_compression/main.py
import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaConfig
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 如果GPU设备可以，则用GPU；否则用CPU设备

class SafeLlamaPixelAR(nn.Module):
    """
    逐像素自回归：
      - 将整张图 (batch, 3, H, W) 展平为 (batch, 3*H*W) 的像素序列，像素值视作 token IDs (0~255)。
      - Llama 的词表大小改为 256，以便直接用像素值做离散分类。
      - 在 Llama 输出之后插入一个可学习的 prob 线性层，
        仅训练该 prob 层和 embedding 层，其余参数冻结。
      - 对模型输出做 shifted one step 的自回归训练。
    """
    def __init__(self, model_path="./saved_model") -> None:
        super().__init__()
        # 1) 载入配置并将 vocab_size 改为 256
        self.config = LlamaConfig.from_pretrained(model_path)
        self.config.vocab_size = 256

        # 2) 载入 Llama 并忽略大小不匹配
        self.llama = LlamaModel.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        # 缩小词表到 256
        self.llama.resize_token_embeddings(256)

        # 3) 冻结 Llama 本体参数，但留出 embedding 部分供训练
        self.embeddings = self.llama.get_input_embeddings()
        self.llama.set_input_embeddings(None)
        for param in self.embeddings.parameters():
            param.requires_grad = True

        # 新增的可学习 prob 层，将 hidden_size 映射至 256
        self.prob = nn.Linear(self.config.hidden_size, 256)

        # 定义交叉熵损失
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data, _):
        """
        data: [batch, 3, H, W], 像素取值 0~255 (bfloat16 或 int)
        返回:
          logits: [batch, seq_len, vocab_size=256]
          bpp: 每像素的平均交叉熵损失
        """
        # 将像素先转换为 int 范围 [0,255]
        pixel_ids = data.long().clamp(0, 255)  # [b, 3, H, W]

        # 展平 -> [b, 3*H*W]
        b, c, h, w = pixel_ids.shape
        seq_len = c * h * w
        pixel_ids = pixel_ids.view(b, seq_len)

        # 调用 Llama 模型，要求输出 hidden_states 用于后续 prob 映射
        outputs = self.llama(
            input_ids=pixel_ids,        # [b, seq_len]
            use_cache=False,
            output_hidden_states=True
        )
        # 取最后一层隐状态，形状 [b, seq_len, hidden_size]
        hidden = outputs.hidden_states[-1]
        # 通过可学习的 prob 层映射，输出 logits: [b, seq_len, 256]
        logits = self.prob(hidden)

        # Shift 操作：移除最后一个时间步的 logits（无下一个像素可预测）
        shifted_logits = logits[:, :-1].contiguous()  # [b, seq_len - 1, 256]
        # 移除第一个像素作为 target（它没有前文）
        shifted_targets = pixel_ids[:, 1:].contiguous()  # [b, seq_len - 1]

        # 计算逐像素交叉熵：loss_fn 会将 [b*(seq_len-1), vocab_size] 与 [b*(seq_len-1)] 做对比
        ce_loss = self.loss_fn(
            shifted_logits.view(-1, 256),
            shifted_targets.view(-1)
        )
        # 转换为 base-2 的交叉熵
        bpp = ce_loss / math.log(2)

        return logits, bpp


# =========== 使用示例 ===========
if __name__ == "__main__":
    # 创造一个 [2, 3, 16, 16] 的假图像（示例中用较小尺寸便于调试）
    dummy_input = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.bfloat16, device=device)

    model = SafeLlamaPixelAR().cuda().bfloat16()
    model.to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, bpp = model(dummy_input, None)

    print("Logits shape:", logits.shape)   # 期望形状 [2, 3*16*16, 256]
    print("Avg BPP value:", bpp.item())