import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torchaudio
import pickle
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor, HubertModel
from tqdm import tqdm  # For progress tracking

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
IEMOCAP_TRAIN_PATH = "/kaggle/input/metadata1/metadata-1/IEMOCAP_metadata_train.csv"
IEMOCAP_VAL_PATH = "/kaggle/input/metadata1/metadata-1/IEMOCAP_metadata_val.csv"
IEMOCAP_TEST_PATH = "/kaggle/input/metadata1/metadata-1/IEMOCAP_metadata_test.csv"

OUTPUT_DIR = "/kaggle/working/"
TOKENIZER = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')
TEXT_MODEL = RobertaModel.from_pretrained('FacebookAI/roberta-base').to(device)

AUDIO_PROCESSOR = Wav2Vec2Processor.from_pretrained('facebook/hubert-large-ls960-ft')
AUDIO_MODEL = HubertModel.from_pretrained('facebook/hubert-large-ls960-ft').to(device)

def process_row(row, tokenizer, text_model, audio_processor, audio_model, device):
    # Process text
    text = row['raw_text']
    text_token = tokenizer(text, return_tensors="pt")
    text_token = text_token.to(device)
    text_outputs = text_model(**text_token)
    text_embeddings = text_outputs.last_hidden_state
    text_embed = text_embeddings[:, 0, :][0].cpu()

    # Process audio
    audio_file = row['audio_file']
    audio_signal, _ = torchaudio.load(audio_file, normalize=True)
    audio_signal = audio_signal.squeeze(0).to(device)  # Ensure it's 1D for HuBERT
    inputs = audio_processor(audio_signal, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    audio_outputs = audio_model(**inputs)
    audio_embed = audio_outputs.last_hidden_state.mean(dim=1).cpu().squeeze()

    # Process label
    label = row['label']
    label = torch.tensor(label)

    return {
        'text_embed': text_embed,
        'audio_embed': audio_embed,
        'label': label
    }

def process_dataset(input_path, output_path, tokenizer, text_model, audio_processor, audio_model, device):
    data_list = pd.read_csv(input_path)
    processed_data = []
    with torch.no_grad():
        for idx, row in tqdm(data_list.iterrows(), total=len(data_list), desc=f"Processing {output_path.split('/')[-1]}"):
            processed_data.append(process_row(row, tokenizer, text_model, audio_processor, audio_model, device))
    with open(output_path, "wb") as f:
        pickle.dump(processed_data, f)
    print(f"Processed data saved to {output_path}")

def main():
    print("Loading models and tokenizer...")
    tokenizer = TOKENIZER
    text_model = TEXT_MODEL
    audio_processor = AUDIO_PROCESSOR
    audio_model = AUDIO_MODEL

    print("Processing training set...")
    process_dataset(
        IEMOCAP_TRAIN_PATH,
        f"{OUTPUT_DIR}IEMOCAP_RoBERTa_HUBERT_train.pkl",
        tokenizer,
        text_model,
        audio_processor,
        audio_model,
        device
    )

    print("Processing validation set...")
    process_dataset(
        IEMOCAP_VAL_PATH,
        f"{OUTPUT_DIR}IEMOCAP_RoBERTa_HUBERT_val.pkl",
        tokenizer,
        text_model,
        audio_processor,
        audio_model,
        device
    )

    print("Processing testing set...")
    process_dataset(
        IEMOCAP_TEST_PATH,
        f"{OUTPUT_DIR}IEMOCAP_RoBERTa_HUBERT_test.pkl",
        tokenizer,
        text_model,
        audio_processor,
        audio_model,
        device
    )

if __name__ == "__main__":
    main()
