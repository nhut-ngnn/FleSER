import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
import wandb
from sklearn.model_selection import KFold
from training.CustomizedDataset import CustomizedDataset
from training.architecture import FlexibleMMSER
from ultis import *

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_metadata = "feature/IEMOCAP_BERT_WAV2VEC_train.pkl"
val_metadata = "feature/IEMOCAP_BERT_WAV2VEC_val.pkl"
test_metadata = "feature/IEMOCAP_BERT_WAV2VEC_test.pkl"

BATCH_SIZE = 128
LEARNING_RATE = 0.0001
NUM_EPOCHS = 150
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
PROJECT_NAME = "IEMOCAP-FlexibleMMSER-Alpha-Experiment-self"
MODEL_NAME = "BERT_WAV2VEC"
FUZZY_METHOD = "self_attention"
DATASET_NAME = "IEMOCAP" 
K_FOLDS = 5
NUM_CLASSES = 4

train_dataset = CustomizedDataset(train_metadata)
val_dataset = CustomizedDataset(val_metadata)
test_dataset = CustomizedDataset(test_metadata)

kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

for alpha in ALPHA_VALUES:
    fold_results = []
    test_results = []

    wandb.init(
        project=PROJECT_NAME,
        name=f"{MODEL_NAME}_{FUZZY_METHOD}_alpha_{alpha}",
        config={
            "name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "model": "FlexibleMMSER",
            "dataset": DATASET_NAME,
            "feature": MODEL_NAME,
            "fuzzy_method": FUZZY_METHOD,
            "alpha": alpha,
            "k_folds": K_FOLDS
        }
    )
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"Fold {fold + 1}/{K_FOLDS} - Training with alpha = {alpha}")
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = FlexibleMMSER(num_classes=NUM_CLASSES).to(device)       # Change num_classes to classses in the dataset
        model.alpha = alpha
        model.fusion_method = FUZZY_METHOD
        print_model_parameters(model)

        save_path = f"saved_models/{DATASET_NAME}_{MODEL_NAME}_fold{fold + 1}_alpha_{alpha}.pt"

        train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            save_path=save_path
        )
        
        val_wa, val_ua, val_wf1, val_uf1, _, _ = model_prediction(model, val_loader, calculate_accuracy)
        fold_results.append((val_wa, val_ua, val_wf1, val_uf1))
        print(f"Fold {fold + 1} Results: WA: {val_wa:.4f}, UA: {val_ua:.4f}, WF1: {val_wf1:.4f}, UF1: {val_uf1:.4f}")
        print("=" * 50)

        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()

        test_wa, test_ua, test_wf1, test_uf1, _, _ = model_prediction(model, test_loader, calculate_accuracy)
        test_results.append((test_wa, test_ua, test_wf1, test_uf1))
        print(f"Fold {fold + 1} Test Results: WA: {test_wa:.4f}, UA: {test_ua:.4f}, WF1: {test_wf1:.4f}, UF1: {test_uf1:.4f}")
        print("=" * 50)

    mean_val_wa = sum([res[0] for res in fold_results]) / K_FOLDS
    mean_val_ua = sum([res[1] for res in fold_results]) / K_FOLDS
    mean_val_wf1 = sum([res[2] for res in fold_results]) / K_FOLDS
    mean_val_uf1 = sum([res[3] for res in fold_results]) / K_FOLDS

    mean_test_wa = sum([res[0] for res in test_results]) / K_FOLDS
    mean_test_ua = sum([res[1] for res in test_results]) / K_FOLDS
    mean_test_wf1 = sum([res[2] for res in test_results]) / K_FOLDS
    mean_test_uf1 = sum([res[3] for res in test_results]) / K_FOLDS

    print(f"Alpha = {alpha} - Validation Results Summary:")
    print(f"Mean WA: {mean_val_wa:.4f}, Mean UA: {mean_val_ua:.4f}, Mean WF1: {mean_val_wf1:.4f}, Mean UF1: {mean_val_uf1:.4f}")
    print("=" * 50)

    print(f"Alpha = {alpha} - Test Results Summary:")
    print(f"Mean WA: {mean_test_wa:.4f}, Mean UA: {mean_test_ua:.4f}, Mean WF1: {mean_test_wf1:.4f}, Mean UF1: {mean_test_uf1:.4f}")
    print("=" * 50)

    wandb.log({
        "alpha": alpha,
        "mean_val_wa": mean_val_wa,
        "mean_val_ua": mean_val_ua,
        "mean_val_wf1": mean_val_wf1,
        "mean_val_uf1": mean_val_uf1,
        "mean_test_wa": mean_test_wa,
        "mean_test_ua": mean_test_ua,
        "mean_test_wf1": mean_test_wf1,
        "mean_test_uf1": mean_test_uf1
    })

    wandb.finish()