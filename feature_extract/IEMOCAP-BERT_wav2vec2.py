import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torchaudio
import pickle
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel, Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm  # For progress tracking

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IEMOCAP_TRAIN_PATH = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/Multimodal-Speech-Emotion-Recognition/metadata/IEMOCAP_metadata_train.csv"
IEMOCAP_VAL_PATH = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/Multimodal-Speech-Emotion-Recognition/metadata/IEMOCAP_metadata_val.csv"
IEMOCAP_TEST_PATH = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/Multimodal-Speech-Emotion-Recognition/metadata/IEMOCAP_metadata_test.csv"

OUTPUT_DIR = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/feature/"

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
TEXT_MODEL = BertModel.from_pretrained('bert-base-uncased').to(device)
WAV2VEC_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
WAV2VEC_MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

def extract_wav2vec_features(audio_file, wav2vec_processor, wav2vec_model, device):
    waveform, sample_rate = torchaudio.load(audio_file)
    
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    input_values = wav2vec_processor(
        waveform.squeeze(), 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values.to(device)
    with torch.no_grad():
        outputs = wav2vec_model(input_values)
        features = outputs.last_hidden_state
        audio_embed = features.mean(dim=1).squeeze().cpu()
    
    return audio_embed

def process_row(row, tokenizer, text_model, wav2vec_processor, wav2vec_model, device):

    text = row['raw_text']
    text_token = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    text_token = {k: v.to(device) for k, v in text_token.items()}
    
    with torch.no_grad():
        text_outputs = text_model(**text_token)
        text_embeddings = text_outputs.last_hidden_state
        text_embed = text_embeddings[:, 0, :][0].cpu()
    
    audio_file = row['audio_file']
    audio_embed = extract_wav2vec_features(
        audio_file, 
        wav2vec_processor, 
        wav2vec_model, 
        device
    )
    
    label = torch.tensor(row['label'])
    
    return {
        'text_embed': text_embed,
        'audio_embed': audio_embed,
        'label': label
    }

def process_dataset(input_path, output_path, tokenizer, text_model, wav2vec_processor, wav2vec_model, device):
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
                    wav2vec_model, 
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
        IEMOCAP_TRAIN_PATH,
        f"{OUTPUT_DIR}IEMOCAP_BERT_WAV2VEC_train.pkl",
        TOKENIZER,
        TEXT_MODEL,
        WAV2VEC_PROCESSOR,
        WAV2VEC_MODEL,
        device
    )

    print("Processing validation set...")
    process_dataset(
        IEMOCAP_VAL_PATH,
        f"{OUTPUT_DIR}IEMOCAP_BERT_WAV2VEC_val.pkl",
        TOKENIZER,
        TEXT_MODEL,
        WAV2VEC_PROCESSOR,
        WAV2VEC_MODEL,
        device
    )

    print("Processing testing set...")
    process_dataset(
        IEMOCAP_TEST_PATH,
        f"{OUTPUT_DIR}IEMOCAP_BERT_WAV2VEC_test.pkl",
        TOKENIZER,
        TEXT_MODEL,
        WAV2VEC_PROCESSOR,
        WAV2VEC_MODEL,
        device
    )

if __name__ == "__main__":
    main()