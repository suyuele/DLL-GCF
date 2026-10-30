import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DynamicRouting(nn.Module):
    def __init__(self, in_dim, out_dim, num_caps, iterations=3):
        super().__init__()
        self.iterations = iterations
        self.num_caps = num_caps
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def squash(self, s, dim=-1):
        norm = torch.norm(s, dim=dim, keepdim=True)
        return (norm / (1 + norm ** 2)) * (s / (norm + 1e-8))

    def forward(self, x):
        # x shape: [B, N_primary, in_dim]
        B, N, _ = x.shape
        u_hat = self.W(x)  # [B, N, out_dim]
        b = torch.zeros(B, N, self.num_caps, device=x.device)

        for i in range(self.iterations):
            c = F.softmax(b, dim=-1)  # [B, N, num_caps]
            s = torch.einsum('bnk,bnd->bkd', c, u_hat)  # [B, num_caps, out_dim]
            v = self.squash(s)

            if i < self.iterations - 1:
                v_expand = v.unsqueeze(1)  # [B, 1, num_caps, out_dim]
                u_expand = u_hat.unsqueeze(2)  # [B, N, 1, out_dim]
                b = b + torch.sum(v_expand * u_expand, dim=-1)
        return v


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x: [B, K, D], adj: [B, K, K]
        support = torch.matmul(x, self.weight)
        output = torch.bmm(adj, support)
        return output


class GraphCapsuleVisualEncoder(nn.Module):
    def __init__(self, num_caps=16, dim_caps=16, hidden_dim=256):
        super().__init__()
        self.num_caps = num_caps
        self.dim_caps = dim_caps

        # 1. Backbone (ResNet-50)
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output: [B, 2048, 7, 7]

        # 2. Primary Capsule
        self.prim_caps = nn.Linear(2048, 8)  # D_prim = 8

        # 3. Routing
        self.routing = DynamicRouting(in_dim=8, out_dim=dim_caps, num_caps=num_caps, iterations=3)

        # 4. Graph Encoder (GCN)
        self.gcn1 = GraphConvLayer(dim_caps, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)

        # 5. Graph Decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_caps)
        )

    def forward(self, x, mask_ratio=0.2):
        B = x.size(0)
        # Backbone
        features = self.backbone(x)  # [B, 2048, 7, 7]
        features = features.view(B, 2048, -1).permute(0, 2, 1)  # [B, 49, 2048]

        # Primary Capsule & Routing
        prim_features = self.prim_caps(features)  # [B, 49, 8]
        caps_out = self.routing(prim_features)  # [B, K, D_caps]

        # Graph Construction (Top-k adjacency)
        # (简化版：这里直接使用特征余弦相似度构建全连接图)
        caps_norm = F.normalize(caps_out, p=2, dim=-1)
        adj = torch.bmm(caps_norm, caps_norm.transpose(1, 2))
        adj = torch.relu(adj)  # Clamp negative
        adj = adj * (1 - torch.eye(self.num_caps, device=x.device).unsqueeze(0))  # Remove self-loop for adjacency

        # Masking (for reconstruction)
        mask = torch.rand(B, self.num_caps, 1, device=x.device) > mask_ratio
        masked_caps = caps_out * mask.float()

        # GCN with symmetric normalization
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        adj_norm = adj / deg
        h1 = F.relu(self.gcn1(masked_caps, adj_norm))
        h2 = self.gcn2(h1, adj_norm)  # [B, K, D_hidden]

        # Decode (Reconstruct masked nodes)
        recon_caps = self.decoder_mlp(h2)  # [B, K, D_caps]

        # GAP for global visual vector
        visual_vec = h2.mean(dim=1)  # [B, D_hidden]

        return caps_out, visual_vec, recon_caps, mask