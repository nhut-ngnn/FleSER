import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, Wav2Vec2Processor, Wav2Vec2Model

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ultis import set_seed

set_seed(42)

NOISE_DIR = "/home/tri.pm/polyp/fptu/MinhNhut/FleSER_private/feature_extract/noisex92/"
NOISE_TYPES = {
    "Babble": torchaudio.load(os.path.join(NOISE_DIR, "babble.wav"))[0],
    "F16": torchaudio.load(os.path.join(NOISE_DIR, "f16.wav"))[0],
    "Volvo": torchaudio.load(os.path.join(NOISE_DIR, "volvo.wav"))[0],
    "HF-Channel": torchaudio.load(os.path.join(NOISE_DIR, "hfchannel.wav"))[0],
    "White": torchaudio.load(os.path.join(NOISE_DIR, "white.wav"))[0],
}

def add_noise_from_file(clean_waveform, noise_waveform, snr_db):
    """
    Mixs clean waveform with preloaded noise at target SNR.
    """
    clean = clean_waveform.numpy()
    noise = noise_waveform.numpy()

    if noise.shape[1] < clean.shape[1]:
        repeat_factor = int(np.ceil(clean.shape[1] / noise.shape[1]))
        noise = np.tile(noise, (1, repeat_factor))
    noise = noise[:, :clean.shape[1]]

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / noise_power)
    scaled_noise = noise * scaling_factor

    noisy = clean + scaled_noise
    return torch.from_numpy(noisy).float()


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
        with torch.no_grad():
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IEMOCAP_TEST_PATH = "metadata/IEMOCAP_metadata_test.csv"
OUTPUT_DIR = "feature/"

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
TEXT_MODEL = BERTEmbeddingModel().to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    TEXT_MODEL = torch.nn.DataParallel(TEXT_MODEL)

bert_checkpoint = torch.load('/home/tri.pm/polyp/fptu/MinhNhut/FleSER_private/fine-tuning/model/IEMOCAP/best_bert_embeddings.pt')
from collections import OrderedDict
state_dict = bert_checkpoint['model_state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if isinstance(TEXT_MODEL, torch.nn.DataParallel):
        if not k.startswith('module.'):
            k = 'module.' + k
    else:
        if k.startswith('module.'):
            k = k[7:]
    new_state_dict[k] = v
TEXT_MODEL.load_state_dict(new_state_dict, strict=False)
TEXT_MODEL.eval()

AUDIO_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
AUDIO_MODEL = AudioEmbeddingModel().to(device)
audio_checkpoint = torch.load('/home/tri.pm/polyp/fptu/MinhNhut/FleSER_private/fine-tuning/model/IEMOCAP/best_wav2vec_embeddings.pt')
AUDIO_MODEL.load_state_dict(audio_checkpoint['model_state_dict'], strict=False)
AUDIO_MODEL.eval()


def extract_text_features(text, tokenizer, text_model, device):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None
    text_token = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    text_token = {k: v.to(device) for k, v in text_token.items()}
    with torch.no_grad():
        text_embed, _ = text_model(
            text_token['input_ids'],
            text_token['attention_mask']
        )
    return text_embed.squeeze().cpu()

def extract_audio_features(audio_file, wav2vec_processor, wav2vec_model, device, snr_db, noise_type):
    waveform, sample_rate = torchaudio.load(audio_file)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if noise_type in NOISE_TYPES:
        noise_waveform = NOISE_TYPES[noise_type]
        waveform = add_noise_from_file(waveform, noise_waveform, snr_db)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    input_values = wav2vec_processor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values.to(device)

    with torch.no_grad():
        audio_embed, _ = wav2vec_model(input_values)
    return audio_embed.squeeze().cpu()

def process_dataset_with_noise(input_path, output_path, snr_db, noise_type):
    data_list = pd.read_csv(input_path)
    processed_data = []
    for idx, row in tqdm(data_list.iterrows(), total=len(data_list), desc=f"{noise_type} {snr_db}dB"):
        text_embed = extract_text_features(row.get('raw_text', ''), TOKENIZER, TEXT_MODEL, device)
        audio_embed = extract_audio_features(row.get('audio_file', ''), AUDIO_PROCESSOR, AUDIO_MODEL, device, snr_db, noise_type)
        label = torch.tensor(row.get('label', -1))
        if text_embed is not None and audio_embed is not None:
            processed_data.append({
                'text_embed': text_embed,
                'audio_embed': audio_embed,
                'label': label
            })
    with open(output_path, "wb") as f:
        pickle.dump(processed_data, f)
    print(f"Saved noisy embeddings to {output_path}")


if __name__ == "__main__":
    for snr_db in [20, 10, 5, 0]:
        for noise_type in ["Babble", "F16", "Volvo", "HF-Channel", "White"]:
            output_path = f"{OUTPUT_DIR}IEMOCAP_BERT_WAV2VEC_test_{noise_type.replace('-', '_')}_{snr_db}dB.pkl"
            process_dataset_with_noise(IEMOCAP_TEST_PATH, output_path, snr_db, noise_type)
