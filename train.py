import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models.dll_gcf import DLL_GCF


# from dataset import MeterDataset # 引入上面的Dataset

def train_model(model, dataloader, optimizer, device, epochs=100, lambda_recon=0.1):
    model.train()
    for epoch in range(epochs):
        total_loss_epoch = 0
        for batch in dataloader:
            img = batch['image'].to(device)
            text_int = batch['text_int'].to(device)
            text_app = batch['text_app'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # Forward pass
            logits, loss, loss_cls, loss_recon = model(img, text_int, text_app, labels)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] - Total Loss: {total_loss_epoch / len(dataloader):.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DLL_GCF(num_classes=4).to(device)

    # 冻结 LLM 参数 (大模型离线生成)
    # 若 RoBERTa 也参与训练，则调整 optimizer 的 params
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 加载数据（此处需替换为真实数据）
    # dataset = MeterDataset(...)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # train_model(model, dataloader, optimizer, device)
    print("Model architecture successfully compiled and ready for training.")