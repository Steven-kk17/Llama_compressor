import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPT2PixelAR(nn.Module):
    """
    GPT2-based 逐像素自回归压缩模型:
      - 将整张图 (batch, 3, H, W) 展平为 (batch, 3*H*W) 的像素序列，像素值视作 token IDs (0~255)
      - GPT2 的词表大小改为 256，以便直接用像素值做离散分类
      - 在 GPT2 输出之后添加一个可学习的 prob 线性层，映射到像素值预测
      - 对模型输出做 shifted one step 的自回归训练
    """
    def __init__(self, model_path="/remote-home/wufeiyang/gpt2_model") -> None:
        super().__init__()
        # 1) 载入配置并将 vocab_size 改为 256
        self.config = GPT2Config.from_pretrained(model_path)
        self.config.vocab_size = 256  # 调整为像素值范围

        # 2) 载入 GPT2 并忽略大小不匹配
        self.gpt2 = GPT2Model.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True
        )
        # 缩小词表到 256
        self.gpt2.resize_token_embeddings(256)

        # 3) 选择性冻结参数 - 只保留embedding层可训练
        # 先冻结所有参数
        for param in self.gpt2.parameters():
            param.requires_grad = True
            
        # # 然后只解冻embedding层
        # for param in self.gpt2.wte.parameters():
        #     param.requires_grad = True

        # 新增的可学习 prob 层，将 hidden_size 映射至 256
        self.prob = nn.Linear(self.config.n_embd, 256)

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

        # 调用 GPT2 模型获取隐藏状态
        outputs = self.gpt2(
            input_ids=pixel_ids,
            output_hidden_states=True
        )
        # 取最后一层隐状态，形状 [b, seq_len, hidden_size]
        hidden = outputs.last_hidden_state
        # 通过可学习的 prob 层映射，输出 logits: [b, seq_len, 256]
        logits = self.prob(hidden)

        # Shift 操作：移除最后一个时间步的 logits（无下一个像素可预测）
        shifted_logits = logits[:, :-1].contiguous()  # [b, seq_len - 1, 256]
        # 移除第一个像素作为 target（它没有前文）
        shifted_targets = pixel_ids[:, 1:].contiguous()  # [b, seq_len - 1]

        # 计算逐像素交叉熵
        ce_loss = self.loss_fn(
            shifted_logits.view(-1, 256),
            shifted_targets.view(-1)
        )
        # 转换为 base-2 的交叉熵
        bpp = ce_loss / math.log(2)

        return logits, bpp
