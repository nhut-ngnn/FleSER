import torch
from torch.utils.data import DataLoader
import wandb
from training.CustomizedDataset import CustomizedDataset
from ultis import *

from training.BERT_ECAPA import FlexibleMMSER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
alpha_values = [0.5]

train_metadata = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/feature/IEMOCAP_RoBERTa_ECAPA_train.pkl"
val_metadata = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/feature/IEMOCAP_RoBERTa_ECAPA_val.pkl"

BATCH_SIZE = 128
train_dataset = CustomizedDataset(train_metadata)
val_dataset = CustomizedDataset(val_metadata)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

for alpha in alpha_values:
    wandb.init(
        project="FlexibleMMSER-Alpha-Experiment",
        config={
            "name": "RoBERTa_ECAPA",
            "batch_size": 128,
            "learning_rate": 0.0001,
            "num_epochs": 200,
            "model": "FlexibleMMSER",
            "dataset": "IEMOCAP",
            "feature": "RoBERTa_ECAPA",
            "alpha": alpha
        }
    )
    
    model = FlexibleMMSER(num_classes=4).to(device)
    model.alpha = alpha  
    print(f"Training with alpha = {alpha}")
    print_model_parameters(model)

    save_path = f"model/IEMOCAP_RoBERTa_ECAPA_alpha_{alpha}.pt"
    
    train_and_evaluate(
        model, train_dataloader, val_dataloader, num_epochs=200, save_path=save_path
    )
    
    wandb.finish()
