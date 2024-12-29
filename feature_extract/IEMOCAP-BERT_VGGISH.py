import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torchaudio
import pickle
import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm import tqdm  
from torchvggish import vggish, vggish_input
from torch.nn.functional import pad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IEMOCAP_TRAIN_PATH = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/Multimodal-Speech-Emotion-Recognition/metadata/IEMOCAP_metadata_train.csv"
IEMOCAP_VAL_PATH = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/Multimodal-Speech-Emotion-Recognition/metadata/IEMOCAP_metadata_val.csv"
IEMOCAP_TEST_PATH = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/Multimodal-Speech-Emotion-Recognition/metadata/IEMOCAP_metadata_test.csv"

OUTPUT_DIR = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/feature/"
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
TEXT_MODEL = BertModel.from_pretrained('bert-base-uncased').to(device)
AUDIO_MODEL = vggish(postprocess=False).to(device)  

def preprocess_audio(audio_file):
    try:
        waveform, sample_rate = torchaudio.load(audio_file)
        if waveform.numel() == 0:
            raise ValueError(f"Audio file {audio_file} is empty.")

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0).unsqueeze(0)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        if waveform.size(1) < 16000:
            padding = 16000 - waveform.size(1)
            waveform = pad(waveform, (0, padding), "constant", 0)

        waveform_np = waveform.squeeze(0).numpy()
        log_mel_spectrogram = vggish_input.waveform_to_examples(waveform_np, sample_rate=16000)
        if log_mel_spectrogram.ndim == 3:  
            log_mel_spectrogram = log_mel_spectrogram.unsqueeze(0)

        return torch.tensor(log_mel_spectrogram)

    except Exception as e:
        raise RuntimeError(f"Error processing audio file {audio_file}: {e}")


def process_row(row, tokenizer, text_model, audio_model, device):
    text = row['raw_text']
    text_token = tokenizer(text, return_tensors="pt")
    text_token = text_token.to(device)
    text_outputs = text_model(**text_token)
    text_embeddings = text_outputs.last_hidden_state
    text_embed = text_embeddings[:, 0, :][0].cpu()

    audio_file = row['audio_file']
    log_mel_spectrogram = preprocess_audio(audio_file)
    audio_embed = audio_model(log_mel_spectrogram.to(device))
    audio_embed = audio_embed.mean(axis=0).cpu()

    label = row['label']
    label = torch.tensor(label)

    return {
        'text_embed': text_embed,
        'audio_embed': audio_embed,
        'label': label
    }

def process_dataset(input_path, output_path, tokenizer, text_model, audio_model, device):
    data_list = pd.read_csv(input_path)
    processed_data = []
    with torch.no_grad():
        for idx, row in tqdm(data_list.iterrows(), total=len(data_list), desc=f"Processing {output_path.split('/')[-1]}"):
            processed_data.append(process_row(row, tokenizer, text_model, audio_model, device))
    with open(output_path, "wb") as f:
        pickle.dump(processed_data, f)
    print(f"Processed data saved to {output_path}")

def main():
    print("Loading models and tokenizer...")
    tokenizer = TOKENIZER
    text_model = TEXT_MODEL
    audio_model = AUDIO_MODEL

    print("Processing training set...")
    process_dataset(
        IEMOCAP_TRAIN_PATH,
        f"{OUTPUT_DIR}IEMOCAP_BERT_VGGish_train.pkl",
        tokenizer,
        text_model,
        audio_model,
        device
    )

    print("Processing validation set...")
    process_dataset(
        IEMOCAP_VAL_PATH,
        f"{OUTPUT_DIR}IEMOCAP_BERT_VGGish_val.pkl",
        tokenizer,
        text_model,
        audio_model,
        device
    )

    print("Processing testing set...")
    process_dataset(
        IEMOCAP_TEST_PATH,
        f"{OUTPUT_DIR}IEMOCAP_BERT_VGGish_test.pkl",
        tokenizer,
        text_model,
        audio_model,
        device
    )

if __name__ == "__main__":
    main()
