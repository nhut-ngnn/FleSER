import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from collections import Counter
import matplotlib.pyplot as plt

from training.CustomizedDataset import CustomizedDataset
from training.FlexibleFuzzyMMSER import FlexibleFuzzyMMSER  # Updated import

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Utility Functions
def calculate_accuracy(y_pred, y_true):
    class_weights = {cls: 1.0 / count for cls, count in Counter(y_true).items()}
    wa = balanced_accuracy_score(y_true, y_pred, sample_weight=[class_weights[cls] for cls in y_true])
    ua = accuracy_score(y_true, y_pred)
    return wa, ua

def train_step(model, dataloader, optim, loss_fn, fuzzy_loss_weight=0.1):
    model.train()
    train_loss, train_wa, train_ua = 0.0, 0.0, 0.0
    for text_embed, audio_embed, label in dataloader:
        text_embed, audio_embed, label = text_embed.to(device), audio_embed.to(device), label.to(device)
        
        # Updated to handle fuzzy output
        y_logits, y_softmax, fuzzy_memberships = model(text_embed, audio_embed)
        y_preds = y_softmax.argmax(dim=1)
        
        wa, ua = calculate_accuracy(y_preds.cpu().numpy(), label.cpu().numpy())
        
        # Standard cross-entropy loss
        cls_loss = loss_fn(y_logits, label)
        
        # Fuzzy membership loss (optional)
        fuzzy_loss = F.mse_loss(fuzzy_memberships, F.one_hot(label, num_classes=fuzzy_memberships.shape[1]).float())
        
        # Combined loss
        total_loss = cls_loss + fuzzy_loss_weight * fuzzy_loss

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        train_loss += total_loss.item()
        train_wa += wa
        train_ua += ua

    return train_loss / len(dataloader), train_wa / len(dataloader), train_ua / len(dataloader)

def eval_step(model, dataloader, loss_fn, fuzzy_loss_weight=0.1):
    model.eval()
    eval_loss, eval_wa, eval_ua = 0.0, 0.0, 0.0
    with torch.no_grad():
        for text_embed, audio_embed, label in dataloader:
            text_embed, audio_embed, label = text_embed.to(device), audio_embed.to(device), label.to(device)
            
            # Updated to handle fuzzy output
            y_logits, y_softmax, fuzzy_memberships = model(text_embed, audio_embed)
            y_preds = y_softmax.argmax(dim=1)
            
            wa, ua = calculate_accuracy(y_preds.cpu().numpy(), label.cpu().numpy())
            
            # Standard cross-entropy loss
            cls_loss = loss_fn(y_logits, label)
            
            # Fuzzy membership loss (optional)
            fuzzy_loss = F.mse_loss(fuzzy_memberships, F.one_hot(label, num_classes=fuzzy_memberships.shape[1]).float())
            
            # Combined loss
            total_loss = cls_loss + fuzzy_loss_weight * fuzzy_loss

            eval_loss += total_loss.item()
            eval_wa += wa
            eval_ua += ua

    return eval_loss / len(dataloader), eval_wa / len(dataloader), eval_ua / len(dataloader)

def train_and_evaluate(model, train_loader, val_loader, num_epochs, lr=0.0001, save_path=None):
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)

    train_loss_hist, val_loss_hist = [], []
    train_wa_hist, val_wa_hist = [], []
    train_ua_hist, val_ua_hist = [], []

    best_val_loss = float('inf')
    best_wa, best_ua = 0.0, 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_wa, train_ua = train_step(model, train_loader, optimizer, criterion)
        val_loss, val_wa, val_ua = eval_step(model, val_loader, criterion)

        lr_scheduler.step()

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_wa_hist.append(train_wa * 100)
        val_wa_hist.append(val_wa * 100)
        train_ua_hist.append(train_ua * 100)
        val_ua_hist.append(val_ua * 100)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wa = val_wa
            best_ua = val_ua
            if save_path:
                torch.save(model.state_dict(), save_path)

        print(f"Train Loss: {train_loss:.4f}, Train WA: {train_wa:.4f}, Train UA: {train_ua:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val WA: {val_wa:.4f}, Val UA: {val_ua:.4f}")
        print("-" * 50)

    print(f"Best Val WA: {best_wa:.4f}, Best Val UA: {best_ua:.4f}")
    return train_loss_hist, val_loss_hist, train_wa_hist, val_wa_hist, train_ua_hist, val_ua_hist

# Existing plotting and dataset loading functions remain the same

# Paths
train_metadata = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/features/IEMOCAP_BERT_ECAPA_train.pkl"
val_metadata = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/features/IEMOCAP_BERT_ECAPA_val.pkl"

# Datasets and Dataloaders
BATCH_SIZE = 128
train_dataset = CustomizedDataset(train_metadata)
val_dataset = CustomizedDataset(val_metadata)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Training
model = FlexibleFuzzyMMSER(num_classes=4).to(device)
save_path = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/saved_models/IEMOCAP_ENG_CMN_FUZZY_BERT_ECAPA.pt"
train_hist, val_hist, train_wa_hist, val_wa_hist, train_ua_hist, val_ua_hist = train_and_evaluate(
    model, train_dataloader, val_dataloader, num_epochs=200, save_path=save_path)

# Plot Metrics
epochs = list(range(1, 201))
# plot_metrics(epochs, train_hist, val_hist, "Loss")
# plot_metrics(epochs, train_wa_hist, val_wa_hist, "Weighted Accuracy")
# plot_metrics(epochs, train_ua_hist, val_ua_hist, "Unweighted Accuracy")