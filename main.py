import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import wandb
from training.CustomizedDataset import CustomizedDataset
# from training.BERT_Wav2Vec import FlexibleMMSER
# from training.BERT_ECAPA import FlexibleMMSER
from training.BERT_HuBERT import FlexibleMMSER
# from training.BERT_VGGish import FlexibleMMSER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Utility Functions
def calculate_accuracy(y_pred, y_true):
    class_weights = {cls: 1.0 / count for cls, count in Counter(y_true).items()}
    wa = balanced_accuracy_score(y_true, y_pred, sample_weight=[class_weights[cls] for cls in y_true])
    ua = accuracy_score(y_true, y_pred)
    return wa, ua

def train_step(model, dataloader, optim, loss_fn):
    model.train()
    train_loss, train_wa, train_ua = 0.0, 0.0, 0.0
    for text_embed, audio_embed, label in dataloader:
        text_embed, audio_embed, label = text_embed.to(device), audio_embed.to(device), label.to(device)
        y_logits, y_softmax = model(text_embed, audio_embed)
        y_preds = y_softmax.argmax(dim=1)
        wa, ua = calculate_accuracy(y_preds.cpu().numpy(), label.cpu().numpy())
        loss = loss_fn(y_logits, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += loss.item()
        train_wa += wa
        train_ua += ua

    return train_loss / len(dataloader), train_wa / len(dataloader), train_ua / len(dataloader)

def eval_step(model, dataloader, loss_fn):
    model.eval()
    eval_loss, eval_wa, eval_ua = 0.0, 0.0, 0.0
    with torch.no_grad():
        for text_embed, audio_embed, label in dataloader:
            text_embed, audio_embed, label = text_embed.to(device), audio_embed.to(device), label.to(device)
            y_logits, y_softmax = model(text_embed, audio_embed)
            y_preds = y_softmax.argmax(dim=1)
            wa, ua = calculate_accuracy(y_preds.cpu().numpy(), label.cpu().numpy())
            loss = loss_fn(y_logits, label)

            eval_loss += loss.item()
            eval_wa += wa
            eval_ua += ua

    return eval_loss / len(dataloader), eval_wa / len(dataloader), eval_ua / len(dataloader)

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
        

def train_and_evaluate(model, train_loader, val_loader, num_epochs, lr=0.0001, save_path=None):
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_val_loss = float('inf')
    best_wa, best_ua = 0.0, 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_wa, train_ua = train_step(model, train_loader, optimizer, criterion)
        val_loss, val_wa, val_ua = eval_step(model, val_loader, criterion)

        lr_scheduler.step()

        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_wa": train_wa * 100,
            "val_wa": val_wa * 100,
            "train_ua": train_ua * 100,
            "val_ua": val_ua * 100,
            "learning_rate": lr_scheduler.get_last_lr()[0],
            "epoch": epoch + 1,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wa = val_wa
            best_ua = val_ua
            if save_path:
                torch.save(model.state_dict(), save_path)
                wandb.save(save_path)  # Save model to wandb

            print(f"Validation loss reduced. Best Val WA: {best_wa:.4f}, Best Val UA: {best_ua:.4f}")
            print("-" * 50)

        print(f"Train Loss: {train_loss:.4f}, Train WA: {train_wa:.4f}, Train UA: {train_ua:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val WA: {val_wa:.4f}, Val UA: {val_ua:.4f}")
        print("-" * 50)

    print(f"Best Val WA: {best_wa:.4f}, Best Val UA: {best_ua:.4f}")
    
def plot_metrics(epochs, train_hist, val_hist, metric_name):
    plt.plot(epochs, train_hist, label=f'Train {metric_name}')
    plt.plot(epochs, val_hist, label=f'Validation {metric_name}')
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f"Training and Validation {metric_name} Over Epochs")
    plt.show()


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