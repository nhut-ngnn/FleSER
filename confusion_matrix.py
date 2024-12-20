import torch
from torch.utils.data import DataLoader
from training.CustomizedDataset import CustomizedDataset
from training.BERT_Wav2Vec import FlexibleMMSER
from ultis import model_prediction, calculate_accuracy
import csv
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
TEST_METADATA_PATH = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/feature/IEMOCAP_BERT_Wav2Vec_test.pkl"
MODEL_BASE_PATH = "model/IEMOCAP_BERT_Wav2Vec_concat_alpha_{}.pt"
ALPHA_VALUES = [1]
NUM_CLASSES = 4 

def load_model(model_path, num_classes, device):
    """Load the model from the given path and move it to the specified device."""
    model = FlexibleMMSER(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    return model

def evaluate_model(model, dataloader, metric_fn):
    """Evaluate the model using the specified dataloader and metric function."""
    return model_prediction(model, dataloader, metric_fn)

def main():
    test_dataset = CustomizedDataset(TEST_METADATA_PATH)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    results = [] 

    for alpha in ALPHA_VALUES:
        model_path = MODEL_BASE_PATH.format(alpha)
        
        try:
            model = load_model(model_path, NUM_CLASSES, DEVICE)
            print(f"Evaluating model with alpha = {alpha}")

            eval_metrics = evaluate_model(model, test_dataloader, calculate_accuracy)
            _, _, _, _, y_true_ls, y_pred_ls = eval_metrics
            
            print(f"Alpha: {alpha}")
            y_true_ls = np.concatenate(y_true_ls, axis=0)
            y_pred_ls = np.concatenate(y_pred_ls, axis=0)
            cm = confusion_matrix(np.array(y_true_ls), np.array(y_pred_ls))
            print(cm)
            
            cmn = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100

            ax = plt.subplots(figsize=(8, 5.5))[1]
            sns.heatmap(cmn, cmap='flare', annot=True, square=True, linecolor='black', linewidths=0.75, ax = ax, fmt = '.2f', annot_kws={'size': 16})
            ax.set_xlabel('Predicted', fontsize=18, fontweight='bold')
            ax.xaxis.set_label_position('bottom')
            ax.xaxis.set_ticklabels(["Anger", "Happiness", "Neutral", "Sadness"], fontsize=16)
            ax.set_ylabel('Ground Truth', fontsize=18, fontweight='bold')
            ax.yaxis.set_ticklabels(["Anger", "Happiness", "Neutral", "Sadness"], fontsize=16)
            plt.tight_layout()
            plt.show()
        except FileNotFoundError:
            print(f"Model file not found for alpha = {alpha}. Skipping...\n")

if __name__ == "__main__":
    main()
