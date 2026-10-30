import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer


class MeterDataset(Dataset):
    def __init__(self, img_paths, texts_int, texts_app, labels, max_len=128):
        self.img_paths = img_paths
        self.texts_int = texts_int
        self.texts_app = texts_app
        self.labels = labels

        # 图像预处理
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 文本 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.max_len = max_len

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Image
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img_tensor = self.transform(img)

        # 2. Texts
        enc_int = self.tokenizer(self.texts_int[idx], padding='max_length', max_length=self.max_len, truncation=True,
                                 return_tensors='pt')
        enc_app = self.tokenizer(self.texts_app[idx], padding='max_length', max_length=self.max_len, truncation=True,
                                 return_tensors='pt')

        input_ids_int = enc_int['input_ids'].squeeze(0)
        mask_int = enc_int['attention_mask'].squeeze(0)
        input_ids_app = enc_app['input_ids'].squeeze(0)
        mask_app = enc_app['attention_mask'].squeeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            'image': img_tensor,
            'text_int': input_ids_int,
            'text_app': input_ids_app,
            'label': label
        }