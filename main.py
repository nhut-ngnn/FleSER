import torch
from torch.utils.data import DataLoader
import wandb
from training.CustomizedDataset import CustomizedDataset
# from training.BERT_HuBERT import FlexibleMMSER
from training.BERT_VGGish import FlexibleMMSER
from ultis import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_metadata = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/feature/IEMOCAP_BERT_VGGish_train.pkl"
val_metadata = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/feature/IEMOCAP_BERT_VGGish_val.pkl"
test_metadata = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/feature/IEMOCAP_BERT_VGGish_test.pkl"
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
PROJECT_NAME = "FlexibleMMSER-Alpha-Experiment"
MODEL_NAME = "BERT_VGGish"
DATASET_NAME = "IEMOCAP"

train_dataset = CustomizedDataset(train_metadata)
val_dataset = CustomizedDataset(val_metadata)
test_dataset = CustomizedDataset(test_metadata)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for alpha in ALPHA_VALUES:
    wandb.init(
        project=PROJECT_NAME,
        config={
            "name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "model": "FlexibleMMSER",
            "dataset": DATASET_NAME,
            "feature": MODEL_NAME,
            "alpha": alpha
        }
    )
    
    model = FlexibleMMSER(num_classes=4).to(device)
    model.alpha = alpha
    print(f"Training with alpha = {alpha}")
    print_model_parameters(model)

    save_path = f"model/{DATASET_NAME}_{MODEL_NAME}_MHA_alpha_{alpha}.pt"
    
    train_and_evaluate(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_epochs=NUM_EPOCHS,
        save_path=save_path
    )
    
    wandb.finish()

