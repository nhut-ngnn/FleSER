import torch
from torch.utils.data import DataLoader

import wandb
from training.CustomizedDataset import CustomizedDataset
from config import *

# from training.BERT_Wav2Vec import FlexibleMMSER
from training.BERT_ECAPA import FlexibleMMSER
# from training.BERT_HuBERT import FlexibleMMSER
# from training.BERT_VGGish import FlexibleMMSER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]

train_metadata = "/kaggle/input/feature/IEMOCAP_BERT_WAV2VEC_train.pkl"
val_metadata = "/kaggle/input/feature/IEMOCAP_BERT_WAV2VEC_val.pkl"

BATCH_SIZE = 128
train_dataset = CustomizedDataset(train_metadata)
val_dataset = CustomizedDataset(val_metadata)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

for alpha in alpha_values:
    wandb.init(
        project="FlexibleMMSER-Alpha-Experiment",
        config={
            "batch_size": 128,
            "learning_rate": 0.0001,
            "num_epochs": 200,
            "model": "FlexibleMMSER",
            "dataset": "IEMOCAP",
            "feature": "BERT-WAV2VEC",
            "alpha": alpha
        }
    )
    
    model = FlexibleMMSER(num_classes=4).to(device)
    model.alpha = alpha  
    print(f"Training with alpha = {alpha}")
    print_model_parameters(model)

    save_path = f"/kaggle/working/IEMOCAP_BERT_WAV2VEC_alpha_{alpha}.pt"
    train_and_evaluate(
        model, train_dataloader, val_dataloader, num_epochs=200, save_path=save_path
    )

    wandb.finish()
