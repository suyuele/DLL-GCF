import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim=256, doc_dim=128, shared_dim=256, num_heads=8, gate_hidden=128):
        super().__init__()
        # Projection Heads
        self.proj_v = nn.Linear(visual_dim, shared_dim)
        self.proj_d = nn.Linear(doc_dim, shared_dim)

        # Multi-head Cross Attention
        self.mha_v2d = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=num_heads, batch_first=True)
        self.mha_d2v = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=num_heads, batch_first=True)

        # Feed Forward Networks
        self.ffn_v = nn.Sequential(nn.Linear(shared_dim, shared_dim * 2), nn.ReLU(),
                                   nn.Linear(shared_dim * 2, shared_dim))
        self.ffn_d = nn.Sequential(nn.Linear(shared_dim, shared_dim * 2), nn.ReLU(),
                                   nn.Linear(shared_dim * 2, shared_dim))

        # Layer Norms
        self.ln_v = nn.LayerNorm(shared_dim)
        self.ln_d = nn.LayerNorm(shared_dim)

        # Modality Confidence Gate
        self.conf_gate = nn.Sequential(
            nn.Linear(shared_dim * 2, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 2)
        )

        # Feature Saliency Gate
        self.sal_gate_v = nn.Linear(shared_dim, shared_dim)
        self.sal_gate_d = nn.Linear(shared_dim, shared_dim)

    def forward(self, V, D):
        # V: [B, K_v, D_h] (Visual tokens)
        # D: [B, K_d, D_d] (Document tokens)

        Z_v = self.proj_v(V)  # [B, K_v, D_s]
        Z_d = self.proj_d(D)  # [B, K_d, D_s]

        # Cross Attention (Document queries Visual)
        attn_out_d, _ = self.mha_d2v(query=Z_d, key=Z_v, value=Z_v)
        u_d = self.ffn_d(self.ln_d(Z_d + attn_out_d))

        # Cross Attention (Visual queries Document)
        attn_out_v, _ = self.mha_v2d(query=Z_v, key=Z_d, value=Z_d)
        u_v = self.ffn_v(self.ln_v(Z_v + attn_out_v))

        # Global Average Pooling (Convert back to vectors for gating)
        u_v_pool = u_v.mean(dim=1)  # [B, D_s]
        u_d_pool = u_d.mean(dim=1)  # [B, D_s]

        # Modality Confidence Gate
        gate_input = torch.cat([u_v_pool, u_d_pool], dim=-1)
        g_v, g_d = torch.chunk(F.softmax(self.conf_gate(gate_input), dim=-1), 2, dim=-1)  # [B, 1] each

        # Feature Saliency Gate
        s_v = torch.sigmoid(self.sal_gate_v(u_v_pool))
        s_d = torch.sigmoid(self.sal_gate_d(u_d_pool))

        # Dual Gating Aggregation
        fused_feature = g_v * (s_v * u_v_pool) + g_d * (s_d * u_d_pool)  # [B, D_s]

        return fused_feature