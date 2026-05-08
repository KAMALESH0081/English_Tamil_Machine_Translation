import torch
import torch.nn as nn
import math

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# -----------------------------
# Multi-Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, seq_len, _ = q.shape

        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # Split heads
        Q = Q.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.to(scores.device)
            scores = scores.masked_fill(
                ~mask,
                torch.finfo(scores.dtype).min
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V

        # Combine heads
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        return self.w_o(out)


# -----------------------------
# Feed Forward
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Encoder Layer
# -----------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


# -----------------------------
# Decoder Layer
# -----------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


# -----------------------------
# Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, src, mask):
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


# -----------------------------
# Decoder
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos(x)

        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)

        return self.fc_out(x)


# -----------------------------
# Full Transformer
# -----------------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, n_heads, d_ff, n_enc, n_dec, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(src_vocab, d_model, n_heads, d_ff, n_enc, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, n_heads, d_ff, n_dec, dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return out

