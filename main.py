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

wandb.init(
    project="FlexibleMMSER", 
    config={
        "batch_size": 128,
        "learning_rate": 0.0001,
        "num_epochs": 200,
        "model": "FlexibleMMSER",
        "dataset": "IEMOCAP",
    }
)
train_metadata = "/kaggle/input/feature/IEMOCAP_BERT_ECAPA_train.pkl"
val_metadata = "/kaggle/input/feature/IEMOCAP_BERT_ECAPA_val.pkl"

# Datasets and Dataloaders
BATCH_SIZE = 128
train_dataset = CustomizedDataset(train_metadata)
val_dataset = CustomizedDataset(val_metadata)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = FlexibleMMSER(num_classes=4).to(device)
print_model_parameters(model)

save_path = "/kaggle/working/ESD_ENG_CMN_BERT_HuBERT.pt"
train_and_evaluate(
    model, train_dataloader, val_dataloader, num_epochs=200, save_path=save_path
)

wandb.finish()