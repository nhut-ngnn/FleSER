import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torchaudio
import pickle
import pandas as pd
from transformers import BertTokenizer, BertModel
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ESD_TRAIN_PATH = "metadata/ESD_metadata_train.csv"
ESD_VAL_PATH = "metadata/ESD_metadata_val.csv"
ESD_TEST_PATH = "metadata/ESD_metadata_test.csv"

OUTPUT_DIR = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/feature/"

TOKENIZER_ENG = BertTokenizer.from_pretrained('bert-base-uncased')
TEXT_MODEL_ENG = BertModel.from_pretrained('bert-base-uncased').to(device)

TOKENIZER_CMN = BertTokenizer.from_pretrained('bert-base-chinese')
TEXT_MODEL_CMN = BertModel.from_pretrained('bert-base-chinese').to(device)

AUDIO_MODEL = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def process_row(row, tokenizer_eng, text_model_eng, tokenizer_cmn, text_model_cmn, audio_model, device):
    text = row['raw_text']
    language = row['language']

    if language == "eng":
        text_token = tokenizer_eng(text, return_tensors="pt").to(device)
        text_outputs = text_model_eng(**text_token)
    elif language == "cmn":
        text_token = tokenizer_cmn(text, return_tensors="pt").to(device)
        text_outputs = text_model_cmn(**text_token)
    else:
        raise ValueError(f"Unsupported language: {language}")

    text_embeddings = text_outputs.last_hidden_state
    text_embed = text_embeddings[:, 0, :][0].cpu()

    audio_file = row['audio_file']
    audio_signal, _ = torchaudio.load(audio_file, normalize=True)
    audio_outputs = audio_model.encode_batch(audio_signal)
    audio_embed = audio_outputs.mean(axis=0)[0]

    label = torch.tensor(row['label'])

    return {
        'text_embed': text_embed,
        'audio_embed': audio_embed,
        'label': label
    }

def process_dataset(input_path, output_path, tokenizer_eng, text_model_eng, tokenizer_cmn, text_model_cmn, audio_model, device):
    data_list = pd.read_csv(input_path)
    processed_data = []
    with torch.no_grad():
        for _, row in tqdm(data_list.iterrows(), total=len(data_list), desc=f"Processing {output_path.split('/')[-1]}"):
            processed_data.append(
                process_row(row, tokenizer_eng, text_model_eng, tokenizer_cmn, text_model_cmn, audio_model, device)
            )
    with open(output_path, "wb") as f:
        pickle.dump(processed_data, f)
    print(f"Processed data saved to {output_path}")

def main():
    print("Loading models and tokenizer...")

    print("Processing training set...")
    process_dataset(
        ESD_TRAIN_PATH,
        f"{OUTPUT_DIR}ECESD_ENG_CMN_BERT_ECAPA_train.pkl",
        TOKENIZER_ENG,
        TEXT_MODEL_ENG,
        TOKENIZER_CMN,
        TEXT_MODEL_CMN,
        AUDIO_MODEL,
        device
    )

    print("Processing validation set...")
    process_dataset(
        ESD_VAL_PATH,
        f"{OUTPUT_DIR}ECESD_ENG_CMN_BERT_ECAPA_val.pkl",
        TOKENIZER_ENG,
        TEXT_MODEL_ENG,
        TOKENIZER_CMN,
        TEXT_MODEL_CMN,
        AUDIO_MODEL,
        device
    )

    print("Processing testing set...")
    process_dataset(
        ESD_TEST_PATH,
        f"{OUTPUT_DIR}ECESD_ENG_CMN_BERT_ECAPA_test.pkl",
        TOKENIZER_ENG,
        TEXT_MODEL_ENG,
        TOKENIZER_CMN,
        TEXT_MODEL_CMN,
        AUDIO_MODEL,
        device
    )

if __name__ == "__main__":
    main()
