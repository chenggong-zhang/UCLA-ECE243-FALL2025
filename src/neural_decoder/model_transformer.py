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
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CNNEmbedding(nn.Module):
    """
    Learnable downsampling front-end.
    Replaces GaussianSmoothing + Unfold.
    Structure: Conv1d -> GeLU -> Conv1d -> GeLU
    """
    def __init__(self, input_dim, hidden_dim, stride_len=4):
        super().__init__()
        
        # We want total stride to be 4 (to match baseline's strideLen=4)
        # Two layers of stride 2 will achieve this.
        
        # Layer 1
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.GELU()
        
        # Layer 2
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


class TransformerEncoderLayerWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, use_rope=True):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if use_rope and self.head_dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even.")
        self.use_rope = use_rope
        self.rope = RotaryPositionalEmbeddings(self.head_dim) if self.use_rope else None

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.scale = self.head_dim ** -0.5

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [T, B, E]
        residual = src
        x = self.norm1(src)
        T, B, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(T, B, self.nhead, self.head_dim)
        k = k.view(T, B, self.nhead, self.head_dim)
        v = v.view(T, B, self.nhead, self.head_dim)

        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        q = q.permute(1, 2, 0, 3)  # [B, H, T, D]
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        attn_weights = self.attn_drop(attn_scores.softmax(dim=-1))
        attn_out = torch.matmul(attn_weights, v)  # [B, H, T, D]

        attn_out = attn_out.permute(2, 0, 1, 3).contiguous().view(T, B, self.d_model)
        attn_out = self.proj_drop(self.out_proj(attn_out))

        x = residual + attn_out

        residual_ff = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual_ff + x
        return x

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim, # This will be d_model (embedding dimension)
        layer_dim,  # Number of Transformer layers
        nDays=24,
        dropout=0.1,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False, # Unused, kept for API compatibility
        nhead=4,             # New param: Number of attention heads
        dim_feedforward=1024, # New param: Internal size of FFN
        use_layer_norm=True,
        use_rope=True,
    ):
        super(TransformerDecoder, self).__init__()

        self.neural_dim = neural_dim
        self.hidden_dim = hidden_dim
        self.nDays = nDays
        self.device = device
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.use_layer_norm = use_layer_norm
        self.use_rope = use_rope
        
        # Day Adaptation (Linear)
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)
        self.inputLayerNonlinearity = torch.nn.Softsign()

        # --- New Learnable Front-end ---
        # Replaces GaussianSmoothing and Unfolder
        # We ignore kernelLen and strideLen args here because the CNN handles it implicitly
        # assuming stride 4 is desired.
        self.cnn_embed = CNNEmbedding(neural_dim, hidden_dim, stride_len=4)
        self.input_layer_norm = nn.LayerNorm(hidden_dim) if self.use_layer_norm else None

        # --- Transformer Specifics ---
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # 3. Transformer Encoder (custom layer to allow RoPE)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithRoPE(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_rope=self.use_rope,
                )
                for _ in range(layer_dim)
            ]
        )

        # 4. Output Projection
        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)
        self.output_layer_norm = nn.LayerNorm(hidden_dim) if self.use_layer_norm else None

    def forward(self, neuralInput, dayIdx):
        # neuralInput: [Batch, Time, Channels]

        # 1. Apply Day Adaptation
        # Day adaptation applies to raw features before any convolution
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # 2. Apply Learnable CNN Embedding (Subsampling)
        # This replaces Unfold + Linear Projection
        # Output: [Batch, NewTime, HiddenDim]
        src = self.cnn_embed(transformedNeural) 
        if self.input_layer_norm is not None:
            src = self.input_layer_norm(src)
        
        # Scale embeddings (Transformer best practice)
        src = src * math.sqrt(self.hidden_dim)

        # 3. Permute for Transformer (SeqLen, Batch, Dim)
        src = src.permute(1, 0, 2)

        # 4. Add Positional Encoding
        src = self.pos_encoder(src)

        # 5. Transformer Layers (with optional RoPE inside attention)
        output = src
        for layer in self.layers:
            output = layer(output)

        # 6. Permute back to (Batch, SeqLen, Dim)
        output = output.permute(1, 0, 2)
        if self.output_layer_norm is not None:
            output = self.output_layer_norm(output)

        # 7. Output Class Probabilities
        seq_out = self.fc_decoder_out(output)


        return seq_out
