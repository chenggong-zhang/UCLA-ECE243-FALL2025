import math
import torch
from torch import nn

from .rope import RotaryPositionalEmbeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class CNNEmbedding(nn.Module):
    """
    Learnable downsampling front-end.
    Replaces GaussianSmoothing + Unfold.
    Structure: Conv1d -> GeLU -> Conv1d -> GeLU
    """

    def __init__(self, input_dim, hidden_dim, stride_len=4):
        super().__init__()

        # Two layers of stride 2 give total stride 4.
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.act2 = nn.GELU()

    def forward(self, x):
        # x: [Batch, Time, Channels] -> [Batch, Channels, Time]
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)

        # [Batch, Hidden, Time] -> [Batch, Time, Hidden]
        x = x.permute(0, 2, 1)
        return x


class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_rope=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RotaryPositionalEmbeddings(d=self.head_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: [S, B, E]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        S, B, _ = q.shape
        q = q.view(S, B, self.num_heads, self.head_dim)
        k = k.view(S, B, self.num_heads, self.head_dim)
        v = v.view(S, B, self.num_heads, self.head_dim)

        if self.use_rope:
            # RoPE expects [S, B, H, D]
            q = self.rope(q)
            k = self.rope(k)

        # [B, H, S, D]
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(torch.bool)  # [B,1,1,S]
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # [B, H, S, D]
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(S, B, self.embed_dim)  # [S, B, E]
        attn_output = self.out_proj(attn_output)
        return attn_output


class TransformerEncoderLayerRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=1024, dropout=0.1, use_rope=False):
        super().__init__()
        self.norm_first = True
        self.self_attn = MultiHeadSelfAttentionRoPE(embed_dim, num_heads, dropout=dropout, use_rope=use_rope)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask)
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(src + self._sa_block(src, src_mask, src_key_padding_mask))
            src = self.norm2(src + self._ff_block(src))
        return src

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,  # d_model
        layer_dim,  # number of Transformer layers
        nDays=24,
        dropout=0.1,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,  # Unused, kept for API compatibility
        nhead=4,
        dim_feedforward=1024,
        use_rope=False,
    ):
        super(TransformerDecoder, self).__init__()

        self.neural_dim = neural_dim
        self.hidden_dim = hidden_dim
        self.nDays = nDays
        self.device = device
        self.strideLen = strideLen
        self.kernelLen = kernelLen

        # Day Adaptation
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)
        self.inputLayerNonlinearity = torch.nn.Softsign()

        self.cnn_embed = CNNEmbedding(neural_dim, hidden_dim, stride_len=4)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # RoPE-capable encoder stack
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerRoPE(
                    embed_dim=hidden_dim,
                    num_heads=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_rope=use_rope,
                )
                for _ in range(layer_dim)
            ]
        )

        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)

    def forward(self, neuralInput, dayIdx):
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        src = self.cnn_embed(transformedNeural)
        src = src * math.sqrt(self.hidden_dim)
        src = src.permute(1, 0, 2)  # [S, B, E]
        src = self.pos_encoder(src)

        for layer in self.layers:
            src = layer(src)

        output = src.permute(1, 0, 2)  # [B, S, E]
        seq_out = self.fc_decoder_out(output)
        return seq_out
