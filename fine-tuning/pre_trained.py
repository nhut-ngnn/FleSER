import argparse
import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer,
    Wav2Vec2Model, Wav2Vec2Processor,
    HubertModel, Wav2Vec2FeatureExtractor
)


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)

        N = z1.size(0)
        z_all = torch.cat([z1_norm, z2_norm], dim=0)
        sim = torch.mm(z_all, z_all.t()) / self.temperature
        sim_mask = torch.eye(2 * N, device=sim.device).bool()
        sim.masked_fill_(sim_mask, -float('inf'))

        labels = torch.arange(N, device=sim.device)
        labels = torch.cat([labels + N, labels])

        loss = F.cross_entropy(sim, labels)
        return loss


class TextDataset(Dataset):
    def __init__(self, metadata_path, tokenizer, max_length=128):
        self.metadata = pd.read_csv(metadata_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.metadata)

    def _augment(self, text):
        words = text.split()
        if len(words) <= 1:
            return text
        dropout_prob = 0.15
        aug = [w for w in words if np.random.rand() > dropout_prob]
        if not aug:
            aug = [words[np.random.randint(len(words))]]
        return ' '.join(aug)

    def __getitem__(self, idx):
        text = self.metadata.iloc[idx]['raw_text']
        aug1 = self._augment(text)
        aug2 = self._augment(text)

        enc1 = self.tokenizer(aug1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        enc2 = self.tokenizer(aug2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids1': enc1['input_ids'].squeeze(),
            'attention_mask1': enc1['attention_mask'].squeeze(),
            'input_ids2': enc2['input_ids'].squeeze(),
            'attention_mask2': enc2['attention_mask'].squeeze()
        }

class BERTEmbeddingModel(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        proj = self.projection(pooled)
        return pooled, proj

class RoBERTaEmbeddingModel(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        proj = self.projection(pooled)
        return pooled, proj


class AudioDataset(Dataset):
    def __init__(self, metadata_path, processor, segment_length=16000):
        self.metadata = pd.read_csv(metadata_path)
        self.processor = processor
        self.segment_length = segment_length

    def __len__(self):
        return len(self.metadata)

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        return waveform.squeeze()

    def _random_segment(self, waveform):
        if waveform.size(0) > self.segment_length:
            start = torch.randint(0, waveform.size(0) - self.segment_length, (1,))
            return waveform[start:start + self.segment_length]
        else:
            return F.pad(waveform, (0, self.segment_length - waveform.size(0)))

    def __getitem__(self, idx):
        path = self.metadata.iloc[idx]['audio_file']
        waveform = self._load_audio(path)
        seg1 = self._random_segment(waveform)
        seg2 = self._random_segment(waveform)

        proc1 = self.processor(seg1.numpy(), sampling_rate=16000, return_tensors='pt', padding=True)
        proc2 = self.processor(seg2.numpy(), sampling_rate=16000, return_tensors='pt', padding=True)

        return {
            'input_values1': proc1['input_values'].squeeze(),
            'input_values2': proc2['input_values'].squeeze()
        }

class Wav2VecEmbeddingModel(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, projection_dim)
        )

    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        pooled = outputs.last_hidden_state.mean(1)
        proj = self.projection(pooled)
        return pooled, proj

class HuBERTEmbeddingModel(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, projection_dim)
        )

    def forward(self, input_values):
        outputs = self.hubert(input_values)
        pooled = outputs.last_hidden_state.mean(1)
        proj = self.projection(pooled)
        return pooled, proj


def train_model(model_name, dataset_name, metadata_path, num_epochs=10, batch_size=32, learning_rate=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dataset = TextDataset(metadata_path, tokenizer)
        model = BERTEmbeddingModel()
    elif model_name == 'RoBERTa':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        dataset = TextDataset(metadata_path, tokenizer)
        model = RoBERTaEmbeddingModel()
    elif model_name == 'Wav2Vec':
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        dataset = AudioDataset(metadata_path, processor)
        model = Wav2VecEmbeddingModel()
    elif model_name == 'HuBERT':
        processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        dataset = AudioDataset(metadata_path, processor)
        model = HuBERTEmbeddingModel()
    else:
        raise ValueError(f"Unsupported model {model_name}")

    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = NTXentLoss()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"{model_name} | {dataset_name} | Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            if 'input_ids1' in batch: 
                ids1 = batch['input_ids1'].to(device)
                mask1 = batch['attention_mask1'].to(device)
                ids2 = batch['input_ids2'].to(device)
                mask2 = batch['attention_mask2'].to(device)
                _, z1 = model(ids1, mask1)
                _, z2 = model(ids2, mask2)
            else:  
                val1 = batch['input_values1'].to(device)
                val2 = batch['input_values2'].to(device)
                _, z1 = model(val1)
                _, z2 = model(val2)

            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"{model_name} | {dataset_name} | Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_dir = f"fine-tuning/model/{dataset_name}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'epoch': epoch
            }, f"{save_dir}/best_{model_name.lower()}_embeddings.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partially fine-tune BERT, RoBERTa, Wav2Vec, HuBERT on SER datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["IEMOCAP", "MELD", "ESD"])
    parser.add_argument("--model", type=str, required=True, choices=["BERT", "RoBERTa", "Wav2Vec", "HuBERT"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)

    args = parser.parse_args()

    dataset_csv_map = {
        "IEMOCAP": "metadata/IEMOCAP_metadata_train.csv",
        "MELD": "metadata/MELD_metadata_train.csv",
        "ESD": "metadata/LOSO_ESD_metadata_train.csv"
    }

    metadata_path = dataset_csv_map[args.dataset]

    train_model(
        model_name=args.model,
        dataset_name=args.dataset,
        metadata_path=metadata_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
