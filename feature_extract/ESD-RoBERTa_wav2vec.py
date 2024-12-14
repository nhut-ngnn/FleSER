import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torchaudio
import pickle
import pandas as pd
from transformers import RobertaTokenizer, AutoTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to dataset and output directory
ESD_TRAIN_PATH = "/kaggle/input/metadata/metadata/ESD_metadata_train.csv"
ESD_VAL_PATH = "/kaggle/input/metadata/metadata/ESD_metadata_val.csv"
ESD_TEST_PATH = "/kaggle/input/metadata/metadata/ESD_metadata_test.csv"
OUTPUT_DIR = "/kaggle/working/"

# Load text models
TOKENIZER_ENG = RobertaTokenizer.from_pretrained('roberta-base')
TEXT_MODEL_ENG = RobertaModel.from_pretrained('roberta-base').to(device)

TOKENIZER_CMN = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
TEXT_MODEL_CMN = RobertaModel.from_pretrained('hfl/chinese-roberta-wwm-ext').to(device)

# Load Wav2Vec2 audio model
WAV2VEC_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
WAV2VEC_MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

def process_row(row, tokenizers, text_models, wav2vec_processor, wav2vec_model, device):
    # Process text based on language
    text = row['raw_text']
    language = row['language']

    if language == 'eng':
        tokenizer = tokenizers['eng']
        text_model = text_models['eng']
    elif language == 'cmn':
        tokenizer = tokenizers['cmn']
        text_model = text_models['cmn']
    else:
        raise ValueError(f"Unsupported language: {language}")

    text_token = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    text_token = {k: v.to(device) for k, v in text_token.items()}
    
    with torch.no_grad():
        text_outputs = text_model(**text_token)
        text_embeddings = text_outputs.last_hidden_state
        text_embed = text_embeddings[:, 0, :][0].cpu()

    # Process audio
    audio_file = row['audio_file']
    audio_signal, sample_rate = torchaudio.load(audio_file, normalize=True)

    # Resample audio if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        audio_signal = resampler(audio_signal)

    inputs = wav2vec_processor(audio_signal.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        audio_outputs = wav2vec_model(**inputs)
        audio_embed = audio_outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()

    # Label
    label = torch.tensor(row['label'])

    return {
        'text_embed': text_embed,
        'audio_embed': audio_embed,
        'label': label
    }

def process_dataset(input_path, output_path, tokenizers, text_models, wav2vec_processor, wav2vec_model, device):
    data_list = pd.read_csv(input_path)
    processed_data = []
    with torch.no_grad():
        for _, row in tqdm(data_list.iterrows(), total=len(data_list), desc=f"Processing {output_path.split('/')[-1]}"):
            if row['language'] in tokenizers:
                processed_data.append(
                    process_row(row, tokenizers, text_models, wav2vec_processor, wav2vec_model, device)
                )
    with open(output_path, "wb") as f:
        pickle.dump(processed_data, f)
    print(f"Processed data saved to {output_path}")

def main():
    print("Loading models and tokenizers...")

    tokenizers = {
        'eng': TOKENIZER_ENG,
        'cmn': TOKENIZER_CMN
    }

    text_models = {
        'eng': TEXT_MODEL_ENG,
        'cmn': TEXT_MODEL_CMN
    }

    print("Processing training set...")
    process_dataset(
        ESD_TRAIN_PATH,
        f"{OUTPUT_DIR}ESD_ENG_CMN_RoBERTa_Wav2Vec_train.pkl",
        tokenizers,
        text_models,
        WAV2VEC_PROCESSOR,
        WAV2VEC_MODEL,
        device
    )

    print("Processing validation set...")
    process_dataset(
        ESD_VAL_PATH,
        f"{OUTPUT_DIR}ESD_ENG_CMN_RoBERTa_Wav2Vec_val.pkl",
        tokenizers,
        text_models,
        WAV2VEC_PROCESSOR,
        WAV2VEC_MODEL,
        device
    )

    print("Processing testing set...")
    process_dataset(
        ESD_TEST_PATH,
        f"{OUTPUT_DIR}ESD_ENG_CMN_RoBERTa_Wav2Vec_test.pkl",
        tokenizers,
        text_models,
        WAV2VEC_PROCESSOR,
        WAV2VEC_MODEL,
        device
    )

if __name__ == "__main__":
    main()
