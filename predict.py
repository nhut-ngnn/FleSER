import torch
from torch.utils.data import Dataset, DataLoader
from training.CustomizedDataset import CustomizedDataset
from training.BERT_ECAPA import FlexibleMMSER
from ultis import model_prediction, calculate_accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
test_metadata_path = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/feature/IEMOCAP_BERT_ECAPA_test.pkl"
model_path = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/model/IEMOCAP/IEMOCAP_BERT_ECAPA.pt"

test_dataset = CustomizedDataset(test_metadata_path)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

loaded_model = FlexibleMMSER(num_classes=4)
loaded_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
loaded_model.to(device)

ec_test_wa, ec_test_ua, y_true_ls, y_pred_ls = model_prediction(loaded_model, test_dataloader, calculate_accuracy)
