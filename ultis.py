import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from collections import Counter
import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data._utils.collate import default_collate

def collate_skip_none(batch):
    """Skip None samples during DataLoader collation."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_pred, y_true):
    class_weights = {cls: 1.0 / count for cls, count in Counter(y_true).items()}
    sample_weights = [class_weights[cls] for cls in y_true]
    ua = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weights)
    wa = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro')
    wf1 = f1_score(y_true, y_pred, average='weighted')
    return wa, ua, mf1, wf1

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

def train_step(model, dataloader, optimizer, loss_fn):
    model.train()
    epoch_loss, epoch_wa, epoch_ua, epoch_mf1, epoch_wf1 = 0, 0, 0, 0, 0

    for text_embed, audio_embed, labels in dataloader:
        text_embed, audio_embed, labels = text_embed.to(device), audio_embed.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(text_embed, audio_embed)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        wa, ua, mf1, wf1 = calculate_metrics(preds, labels_np)

        epoch_loss += loss.item()
        epoch_wa += wa
        epoch_ua += ua
        epoch_mf1 += mf1
        epoch_wf1 += wf1

    n = len(dataloader)
    return epoch_loss / n, epoch_wa / n, epoch_ua / n, epoch_mf1 / n, epoch_wf1 / n

def eval_step(model, dataloader, loss_fn):
    model.eval()
    epoch_loss, epoch_wa, epoch_ua, epoch_mf1, epoch_wf1 = 0, 0, 0, 0, 0

    with torch.no_grad():
        for text_embed, audio_embed, labels in dataloader:
            text_embed, audio_embed, labels = text_embed.to(device), audio_embed.to(device), labels.to(device)
            logits = model(text_embed, audio_embed)
            loss = loss_fn(logits, labels)

            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            wa, ua, mf1, wf1 = calculate_metrics(preds, labels_np)

            epoch_loss += loss.item()
            epoch_wa += wa
            epoch_ua += ua
            epoch_mf1 += mf1
            epoch_wf1 += wf1

    n = len(dataloader)
    return epoch_loss / n, epoch_wa / n, epoch_ua / n, epoch_mf1 / n, epoch_wf1 / n

def compute_soft_class_weight(labels, smoothing=1.2):
    label_freq = Counter(labels)
    total = sum(label_freq.values())
    weights = []
    for i in sorted(label_freq.keys()):
        freq = label_freq[i] / total
        weights.append(1.0 / np.log(smoothing + freq))
    return torch.tensor(weights, dtype=torch.float)
    
def train_and_evaluate(model, train_loader, val_loader, num_epochs, lr=1e-4, save_path=None):
    all_train_labels = []
    for _, _, labels in train_loader:
        all_train_labels.extend(labels.cpu().numpy())

    class_weights_tensor = compute_soft_class_weight(all_train_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=30, verbose=True)

    best_val_loss = float('inf')
    best_wa, best_ua, best_mf1, best_wf1 = 0, 0, 0, 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_wa, train_ua, train_mf1, train_wf1 = train_step(model, train_loader, optimizer, criterion)
        val_loss, val_wa, val_ua, val_mf1, val_wf1 = eval_step(model, val_loader, criterion)

        scheduler.step(val_loss)
        
        if val_wa > best_wa:
            best_wa = val_wa
            best_val_loss, best_ua, best_mf1, best_wf1 = val_loss, val_ua, val_mf1, val_wf1

            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at epoch {epoch} to {save_path}")

            print(f"[Epoch {epoch}] Validation improved - WA: {best_wa:.4f}, UA: {best_ua:.4f}, MF1: {best_mf1:.4f}, WF1: {best_wf1:.4f}")

        print(f"Train Loss: {train_loss:.4f} | WA: {train_wa:.4f}, UA: {train_ua:.4f}, MF1: {train_mf1:.4f}, WF1: {train_wf1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | WA: {val_wa:.4f}, UA: {val_ua:.4f}, MF1: {val_mf1:.4f}, WF1: {val_wf1:.4f}")

    print(f"\nBest Validation WA: {best_wa:.4f}, UA: {best_ua:.4f}, MF1: {best_mf1:.4f}, WF1: {best_wf1:.4f}")

def model_prediction(model, dataloader, metrics_fn):
    model.eval()
    eval_wa, eval_ua, eval_wf1, eval_mf1 = 0, 0, 0, 0
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for text_embed, audio_embed, labels in dataloader:
            text_embed, audio_embed, labels = text_embed.to(device), audio_embed.to(device), labels.to(device)
            logits = model(text_embed, audio_embed)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            wa, ua, mf1, wf1 = metrics_fn(preds, labels_np)

            eval_wa += wa
            eval_ua += ua
            eval_mf1 += mf1
            eval_wf1 += wf1

            y_true_all.extend(labels_np)
            y_pred_all.extend(preds)

    n = len(dataloader)
    eval_wa /= n
    eval_ua /= n
    eval_mf1 /= n
    eval_wf1 /= n

    print(f"\nTest Results - WA: {eval_wa:.4f}, UA: {eval_ua:.4f}, MF1: {eval_mf1:.4f}, WF1: {eval_wf1:.4f}")

    return eval_wa, eval_ua, eval_mf1, eval_wf1, y_true_all, y_pred_all
    
    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(8, 7))
    sns.set(font_scale=1.4)
    ax = sns.heatmap(
        cm_norm,
        annot=True, fmt=".1f", cmap="BuPu",
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 14, "weight": "bold"},
        cbar_kws={"label": "Accuracy (%)"}
    )

    for i in range(len(class_names)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=2))

    ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

