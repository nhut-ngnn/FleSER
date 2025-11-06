import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
import pickle
import pandas as pd
from transformers import BertTokenizer, BertModel, Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import numpy as np
import time

import os
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Models ====================

class BERTEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        projection = self.projection(pooled)
        return pooled, projection

class AudioEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=512):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
        projection = self.projection(pooled)
        return pooled, projection

# ==================== Utility Functions ====================

def measure_params(models_dict):
    total_params = 0
    for name, model in models_dict.items():
        print(f"\n{name} Parameters:")
        print(parameter_count_table(model))
        params = sum(p.numel() for p in model.parameters())
        total_params += params
    print(f"\nTotal Parameters: {total_params/1e6:.3f} M")
    return total_params

def measure_flops(model, inputs):
    flops = FlopCountAnalysis(model, inputs)
    gflops = flops.total() / 1e9
    print(f"FLOPs: {gflops:.4f} GFLOPs")
    return gflops

def measure_latency(forward_fn, repetitions=20):
    timings = []
    with torch.no_grad():
        for _ in range(repetitions):
            start = time.time()
            forward_fn()
            end = time.time()
            timings.append((end - start) * 1000)
    mean_latency = np.mean(timings)
    print(f"Average Latency per inference: {mean_latency:.3f} ms")
    return mean_latency

# ==================== Main Feature Extraction ====================
def extract_features_and_save(input_csv, output_pkl, tokenizer, text_model, processor, audio_model, device):
    df = pd.read_csv(input_csv)
    extracted_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {output_pkl}"):
        raw_text = row['raw_text']
        audio_file = row['audio_file']
        label = int(row['label'])

        # === Text Feature ===
        text_token = tokenizer(raw_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        text_token = {k: v.to(device) for k, v in text_token.items()}

        try:
            with torch.no_grad():
                text_embed, _ = text_model(text_token['input_ids'], text_token['attention_mask'])
            text_embed = text_embed.squeeze(0).cpu()
        except Exception as e:
            print(f"[Skip] Text processing failed at index {idx} ({audio_file}): {e}")
            continue

        # === Audio Feature ===
        try:
            waveform, sr = torchaudio.load(audio_file)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            input_values = processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values.to(device)

            with torch.no_grad():
                audio_embed, _ = audio_model(input_values)
            audio_embed = audio_embed.squeeze(0).cpu()

        except Exception as e:
            print(f"[Skip] Audio file error at index {idx} ({audio_file}): {e}")
            continue

        extracted_data.append({
            "text_embed": text_embed,
            "audio_embed": audio_embed,
            "label": torch.tensor(label, dtype=torch.long)
        })

    with open(output_pkl, "wb") as f:
        pickle.dump(extracted_data, f)
    print(f"Saved extracted features to {output_pkl} (total {len(extracted_data)} samples)")

# ==================== Main ====================

def main():
    ESD_TRAIN_PATH = "metadata/ESD_metadata_train.csv"
    ESD_VAL_PATH = "metadata/ESD_metadata_val.csv"
    ESD_TEST_PATH = "metadata/ESD_metadata_test.csv"
    OUTPUT_DIR = "feature/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    # === Load BERT Model ===
    text_model = BERTEmbeddingModel().to(device)
    bert_ckpt = torch.load('/home/tri.pm/polyp/fptu/MinhNhut/model/ESD/best_bert_embeddings.pt', map_location=device)
    new_state_dict = OrderedDict()
    for k, v in bert_ckpt['model_state_dict'].items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    text_model.load_state_dict(new_state_dict, strict=False)
    text_model.eval()

    # === Load Wav2Vec2 Model ===
    audio_model = AudioEmbeddingModel().to(device)
    wav_ckpt = torch.load('/home/tri.pm/polyp/fptu/MinhNhut/model/ESD/best_wav2vec_embeddings.pt', map_location=device)
    audio_model.load_state_dict(wav_ckpt['model_state_dict'], strict=False)
    audio_model.eval()

    # === Measure Params ===
    print("\n=== Parameter Count ===")
    measure_params({"BERTEmbeddingModel": text_model})
    measure_params({"AudioEmbeddingModel": audio_model})
    # === Measure FLOPs & Latency ===
    print("\n=== FLOPs Measurement ===")
    dummy_text = tokenizer("hello world", return_tensors="pt", padding=True, truncation=True).to(device)
    measure_flops(text_model, (dummy_text['input_ids'], dummy_text['attention_mask']))

    waveform = torch.randn(1, 16000).to(device)
    measure_flops(audio_model, (waveform,))

    print("\n=== Latency Measurement ===")
    measure_latency(lambda: text_model(dummy_text['input_ids'], dummy_text['attention_mask']))
    measure_latency(lambda: audio_model(waveform))

    # === Extract and Save ===
    print("\n=== Extracting Features ===")
    extract_features_and_save(ESD_TRAIN_PATH, f"{OUTPUT_DIR}ESD_BERT_WAV2VEC_train.pkl",
                              tokenizer, text_model, processor, audio_model, device)
    extract_features_and_save(ESD_VAL_PATH, f"{OUTPUT_DIR}ESD_BERT_WAV2VEC_val.pkl",
                              tokenizer, text_model, processor, audio_model, device)
    extract_features_and_save(ESD_TEST_PATH, f"{OUTPUT_DIR}ESD_BERT_WAV2VEC_test.pkl",
                              tokenizer, text_model, processor, audio_model, device)

if __name__ == "__main__":
    main()
