import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torchaudio
import pickle
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2FeatureExtractor, HubertModel
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ultis import set_seed

set_seed(42)

class RoBERTaEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=256):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('FacebookAI/roberta-base')
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        projection = self.projection(pooled)
        return pooled, projection

class AudioEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=256):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, projection_dim)
        )
        
    def forward(self, input_values):
        outputs = self.hubert(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
        projection = self.projection(pooled)
        return pooled, projection
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MELD_TRAIN_PATH = "/home/nhut-minh-nguyen/Documents/FuzzyFusion-SER/FlexibleMMSER/metadata/MELD_metadata_train.csv"
MELD_VAL_PATH = "/home/nhut-minh-nguyen/Documents/FuzzyFusion-SER/FlexibleMMSER/metadata/MELD_metadata_val.csv"
MELD_TEST_PATH = "/home/nhut-minh-nguyen/Documents/FuzzyFusion-SER/FlexibleMMSER/metadata/MELD_metadata_test.csv"

OUTPUT_DIR = "/home/nhut-minh-nguyen/Documents/FuzzyFusion-SER/FlexibleMMSER/feature/"

TOKENIZER = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')
TEXT_MODEL = RoBERTaEmbeddingModel().to(device)
TEXT_MODEL.eval()


WAV2VEC_PROCESSOR = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
HUBERT_MODEL = AudioEmbeddingModel().to(device)
checkpoint = torch.load('fine_tuning/model/MELD/best_hubert_embeddings.pt')
HUBERT_MODEL.load_state_dict(checkpoint['model_state_dict'], strict=False)
HUBERT_MODEL.eval()

def extract_text_features(text, tokenizer, text_model, device):
    try:
        text_token = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        text_token = {k: v.to(device) for k, v in text_token.items()}

        with torch.no_grad():
            text_embed, _ = text_model(
                text_token['input_ids'],
                text_token['attention_mask']
            )
        return text_embed.squeeze().cpu()
    except Exception as e:
        print(f"Error extracting text features: {e}")
        return None
    
def extract_audio_features(audio_file, wav2vec_processor, hubert_model, device):
    try:
        waveform, sample_rate = torchaudio.load(audio_file)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        input_values = wav2vec_processor(
            waveform.squeeze(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.to(device)

        with torch.no_grad():
            embeddings, _ = hubert_model(input_values)
        return embeddings.squeeze().cpu()
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

def process_row(row, tokenizer, text_model, wav2vec_processor, hubert_model, device):
    try:
        text_embed = extract_text_features(row['raw_text'], tokenizer, text_model, device)
        audio_embed = extract_audio_features(row['audio_file'], wav2vec_processor, hubert_model, device)
        label = torch.tensor(row['label'])

        return {
            'text_embed': text_embed,
            'audio_embed': audio_embed,
            'label': label
        }
    except Exception as e:
        print(f"Error processing row: {e}")
        return None
    
def process_dataset(input_path, output_path, tokenizer, text_model, wav2vec_processor, hubert_model, device):
    data_list = pd.read_csv(input_path)
    processed_data = []
    
    with torch.no_grad():
        for idx, row in tqdm(data_list.iterrows(), total=len(data_list), desc=f"Processing {output_path.split('/')[-1]}"):
            try:
                processed_row = process_row(
                    row, 
                    tokenizer, 
                    text_model, 
                    wav2vec_processor, 
                    hubert_model, 
                    device
                )
                processed_data.append(processed_row)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
    
    with open(output_path, "wb") as f:
        pickle.dump(processed_data, f)
    
    print(f"Processed data saved to {output_path}")

def main():
    print("Loading models and tokenizer...")
    
    print("Processing training set...")
    process_dataset(
        MELD_TRAIN_PATH,
        f"{OUTPUT_DIR}MELD_RoBERTa_HUBERT_train.pkl",
        TOKENIZER,
        TEXT_MODEL,
        WAV2VEC_PROCESSOR,
        HUBERT_MODEL,
        device
    )

    print("Processing validation set...")
    process_dataset(
        MELD_VAL_PATH,
        f"{OUTPUT_DIR}MELD_RoBERTa_HUBERT_val.pkl",
        TOKENIZER,
        TEXT_MODEL,
        WAV2VEC_PROCESSOR,
        HUBERT_MODEL,
        device
    )

    print("Processing testing set...")
    process_dataset(
        MELD_TEST_PATH,
        f"{OUTPUT_DIR}MELD_RoBERTa_HUBERT_test.pkl",
        TOKENIZER,
        TEXT_MODEL,
        WAV2VEC_PROCESSOR,
        HUBERT_MODEL,
        device
    )

if __name__ == "__main__":
    main()
