import torch
from torch.utils.data import Dataset, DataLoader
from training.CustomizedDataset import CustomizedDataset
# from training.BERT_ECAPA import FlexibleMMSER
from training.BERT_Wav2Vec import FlexibleMMSER
# from training.BERT_VGGish import FlexibleMMSER

from ultis import model_prediction, calculate_accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
test_metadata_path = "C:/Users/admin/Documents/FuzzyMachineLearning/mymodel/feature/IEMOCAP_BERT_Wav2Vec_test.pkl"
model_path = "model/IEMOCAP_BERT_Wav2Vec_alpha_0.5.pt"
test_dataset = CustomizedDataset(test_metadata_path)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

loaded_model = FlexibleMMSER(num_classes=4)
loaded_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
loaded_model.to(device)

eval_wa, eval_ua, eval_wf1, eval_uf1, y_true_ls, y_pred_ls = model_prediction(loaded_model, test_dataloader, calculate_accuracy)
