import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torchaudio
import pickle
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor, HubertModel
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to dataset and output directory
ESD_TRAIN_PATH = "/kaggle/input/metadata/metadata/ESD_metadata_train.csv"
ESD_VAL_PATH = "/kaggle/input/metadata/metadata/ESD_metadata_val.csv"
ESD_TEST_PATH = "/kaggle/input/metadata/metadata/ESD_metadata_test.csv"
OUTPUT_DIR = "/kaggle/working/"

# Load text models
TOKENIZER_ENG = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')
TEXT_MODEL_ENG = RobertaModel.from_pretrained('FacebookAI/roberta-base').to(device)


# Load HuBERT audio model
HUBERT_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
HUBERT_MODEL = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)

def process_row(row, tokenizer_eng, text_model_eng, hubert_processor, hubert_model, device):
    # Process text
    text = row['raw_text']
    language = row['language']

    if language == "eng":
        text_token = tokenizer_eng(text, return_tensors="pt").to(device)
        text_outputs = text_model_eng(**text_token)
       
    
        text_embeddings = text_outputs.last_hidden_state
        text_embed = text_embeddings[:, 0, :][0].cpu()
    
        # Process audio
        audio_file = row['audio_file']
        audio_signal, sample_rate = torchaudio.load(audio_file, normalize=True)
        audio_signal = audio_signal.to(device)
    
        # Resample audio if necessary
        if sample_rate != hubert_processor.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, 
                                                       new_freq=hubert_processor.feature_extractor.sampling_rate).to(device)
            audio_signal = resampler(audio_signal)
    
        inputs = hubert_processor(audio_signal.squeeze(0), sampling_rate=hubert_processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True).to(device)
        audio_outputs = hubert_model(**inputs)
        audio_embed = audio_outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
    
        # Label
        label = torch.tensor(row['label'])
    
        return {
            'text_embed': text_embed,
            'audio_embed': audio_embed,
            'label': label
        }

def process_dataset(input_path, output_path, tokenizer_eng, text_model_eng, hubert_processor, hubert_model, device):
    data_list = pd.read_csv(input_path)
    processed_data = []
    with torch.no_grad():
        for _, row in tqdm(data_list.iterrows(), total=len(data_list), desc=f"Processing {output_path.split('/')[-1]}"):
            processed_data.append(
                process_row(row, tokenizer_eng, text_model_eng, hubert_processor, hubert_model, device)
            )
    with open(output_path, "wb") as f:
        pickle.dump(processed_data, f)
    print(f"Processed data saved to {output_path}")

def main():
    print("Loading models and tokenizer...")

    print("Processing training set...")
    process_dataset(
        ESD_TRAIN_PATH,
        f"{OUTPUT_DIR}ESD_ENG_RoBERTa_HuBERT_train.pkl",
        TOKENIZER_ENG,
        TEXT_MODEL_ENG,
        HUBERT_PROCESSOR,
        HUBERT_MODEL,
        device
    )

    print("Processing validation set...")
    process_dataset(
        ESD_VAL_PATH,
        f"{OUTPUT_DIR}ESD_ENG_RoBERTa_HuBERT_val.pkl",
        TOKENIZER_ENG,
        TEXT_MODEL_ENG,
        HUBERT_PROCESSOR,
        HUBERT_MODEL,
        device
    )

    print("Processing testing set...")
    process_dataset(
        ESD_TEST_PATH,
        f"{OUTPUT_DIR}ESD_ENG_RoBERTa_HuBERT_test.pkl",
        TOKENIZER_ENG,
        TEXT_MODEL_ENG,
        HUBERT_PROCESSOR,
        HUBERT_MODEL,
        device
    )

if __name__ == "__main__":
    main()