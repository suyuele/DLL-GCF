import torch
import torch.nn as nn
from .visual_encoder import GraphCapsuleVisualEncoder
from .document_encoder import DualSourceDocumentEncoder
from .fusion_module import CrossAttentionFusion


class DLL_GCF(nn.Module):
    def __init__(self, num_classes=4, lambda_recon=0.1):
        super().__init__()
        self.lambda_recon = lambda_recon

        self.visual_encoder = GraphCapsuleVisualEncoder()
        self.document_encoder = DualSourceDocumentEncoder()
        self.fusion_module = CrossAttentionFusion()

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, img, text_int, text_app, labels=None):
        # 1. Visual Encoding
        raw_caps, v_vec, recon_caps, mask = self.visual_encoder(img)

        # 2. Document Encoding
        doc_caps, d_vec = self.document_encoder(text_int, text_app)

        # 3. Cross-Modal Fusion (Use sequences for attention)
        fused_feature = self.fusion_module(raw_caps, doc_caps)

        # 4. Classification
        logits = self.classifier(fused_feature)

        if labels is not None:
            # Classification Loss
            loss_cls = self.ce_loss(logits, labels)

            # Graph Reconstruction Loss (Masked nodes only)
            mask_expanded = mask.expand(-1, -1, raw_caps.size(-1))
            loss_recon = F.mse_loss(recon_caps[mask_expanded], raw_caps[mask_expanded])

            # Total Loss
            total_loss = loss_cls + self.lambda_recon * loss_recon
            return logits, total_loss, loss_cls, loss_recon

        return logits