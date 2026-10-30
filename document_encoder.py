import torch
import torch.nn as nn
from transformers import RobertaModel
from .visual_encoder import DynamicRouting


class DualSourceDocumentEncoder(nn.Module):
    def __init__(self, num_caps=8, dim_caps=128, tensor_shape=(8, 8, 64)):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        roberta_dim = self.roberta.config.hidden_size  # 768

        self.tensor_shape = tensor_shape
        self.linear_proj = nn.Linear(roberta_dim * 2, tensor_shape[0] * tensor_shape[1] * tensor_shape[2])

        self.routing = DynamicRouting(in_dim=tensor_shape[2], out_dim=dim_caps, num_caps=num_caps, iterations=3)

    def forward(self, text_internal, text_appearance):
        # text_* shape: [B, Seq_Len]
        out_int = self.roberta(text_internal).last_hidden_state[:, 0, :]  # [B, 768] (CLS)
        out_app = self.roberta(text_appearance).last_hidden_state[:, 0, :]  # [B, 768]

        # Parallel Concatenation
        concat_features = torch.cat([out_int, out_app], dim=-1)  # [B, 1536]

        # 3D Reshaping
        reshaped = self.linear_proj(concat_features)  # [B, 8*8*64]
        reshaped = reshaped.view(-1, *self.tensor_shape)  # [B, 8, 8, 64]

        # Flatten to sequence for routing
        B = reshaped.size(0)
        prim_docs = reshaped.view(B, self.tensor_shape[0] * self.tensor_shape[1], self.tensor_shape[2])  # [B, 64, 64]

        # Document Capsule Network
        doc_caps = self.routing(prim_docs)  # [B, K', D_caps]

        # GAP for global document vector
        doc_vec = doc_caps.mean(dim=1)  # [B, D_caps]

        return doc_caps, doc_vec