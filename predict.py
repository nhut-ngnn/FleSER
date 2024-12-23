import torch
from torch.utils.data import DataLoader
from training.CustomizedDataset import CustomizedDataset
from training.BERT_ECAPA import FlexibleMMSER
from ultis import model_prediction, calculate_accuracy
import csv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
TEST_METADATA_PATH = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/feature/IEMOCAP_BERT_ECAPA_test.pkl"
MODEL_BASE_PATH = "model/IEMOCAP_BERT_ECAPA_concat_alpha_{}.pt"
ALPHA_VALUES = [0.5]
NUM_CLASSES = 4 
RESULTS_CSV_PATH = "performance-analysics/IEMOCAP_BERT_ECAPA_concat.csv"

def load_model(model_path, num_classes, device):
    """Load the model from the given path and move it to the specified device."""
    model = FlexibleMMSER(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    return model

def evaluate_model(model, dataloader, metric_fn):
    """Evaluate the model using the specified dataloader and metric function."""
    return model_prediction(model, dataloader, metric_fn)

def save_results_to_csv(results, file_path):
    """Save evaluation results to a CSV file."""
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Alpha", "WA", "UA", "WF1", "UF1"])
        writer.writerows(results)

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
            eval_wa, eval_ua, eval_wf1, eval_uf1, _, _ = eval_metrics
            
            print(f"Alpha: {alpha}")
            print(f"  Weighted Accuracy (WA): {eval_wa:.4f}")
            print(f"  Unweighted Accuracy (UA): {eval_ua:.4f}")
            print(f"  Weighted F1-Score (WF1): {eval_wf1:.4f}")
            print(f"  Unweighted F1-Score (UF1): {eval_uf1:.4f}\n")
        except FileNotFoundError:
            print(f"Model file not found for alpha = {alpha}. Skipping...\n")

    save_results_to_csv(results, RESULTS_CSV_PATH)
    print(f"Results saved to {RESULTS_CSV_PATH}")

if __name__ == "__main__":
    main()
