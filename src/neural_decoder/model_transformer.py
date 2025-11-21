import math
import torch
from torch import nn
from .augmentations import GaussianSmoothing

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
        dim_feedforward=1024 # New param: Internal size of FFN
    ):
        super(TransformerDecoder, self).__init__()

        self.neural_dim = neural_dim
        self.hidden_dim = hidden_dim
        self.nDays = nDays
        self.device = device
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        
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

        # --- Transformer Specifics ---
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=True # Pre-Norm
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=layer_dim)

        # 4. Output Projection
        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)

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
        
        # Scale embeddings (Transformer best practice)
        src = src * math.sqrt(self.hidden_dim)

        # 3. Permute for Transformer (SeqLen, Batch, Dim)
        src = src.permute(1, 0, 2)

        # 4. Add Positional Encoding
        src = self.pos_encoder(src)

        # 5. Transformer Layers
        output = self.transformer_encoder(src)

        # 6. Permute back to (Batch, SeqLen, Dim)
        output = output.permute(1, 0, 2)

        # 7. Output Class Probabilities
        seq_out = self.fc_decoder_out(output)
        
        return seq_out
