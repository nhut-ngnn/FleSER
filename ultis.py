from collections import Counter
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
import torch
from torch import nn, optim
import wandb

import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import random
import numpy as np

import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def calculate_metrics(y_pred, y_true):
    class_weights = {cls: 1.0 / count for cls, count in Counter(y_true).items()}
    wa = balanced_accuracy_score(y_true, y_pred, sample_weight=[class_weights[cls] for cls in y_true])
    ua = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro')
    wf1 = f1_score(y_true, y_pred, average='weighted')
    return wa, ua, mf1, wf1

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

# Training step
def train_step(model, dataloader, optim, loss_fn):
    model.train()
    train_loss, train_wa, train_ua, train_mf1, train_wf1 = 0.0, 0.0, 0.0, 0.0, 0.0
    
    for text_embed, audio_embed, label in dataloader:
        text_embed, audio_embed, label = text_embed.to(device), audio_embed.to(device), label.to(device)
        y_logits, y_softmax = model(text_embed, audio_embed)
        y_preds = y_softmax.argmax(dim=1)

        wa, ua, mf1, wf1 = calculate_metrics(y_preds.cpu().numpy(), label.cpu().numpy())
        loss = loss_fn(y_logits, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += loss.item()
        train_wa += wa
        train_ua += ua
        train_mf1 += mf1
        train_wf1 += wf1

    return (train_loss / len(dataloader), 
            train_wa / len(dataloader), 
            train_ua / len(dataloader), 
            train_mf1 / len(dataloader), 
            train_wf1 / len(dataloader))
    
def eval_step(model, dataloader, loss_fn):
    model.eval()
    eval_loss, eval_wa, eval_ua, eval_mf1, eval_wf1 = 0.0, 0.0, 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for text_embed, audio_embed, label in dataloader:
            text_embed, audio_embed, label = text_embed.to(device), audio_embed.to(device), label.to(device)
            y_logits, y_softmax = model(text_embed, audio_embed)
            y_preds = y_softmax.argmax(dim=1)

            wa, ua, mf1, wf1 = calculate_metrics(y_preds.cpu().numpy(), label.cpu().numpy())
            loss = loss_fn(y_logits, label)

            eval_loss += loss.item()
            eval_wa += wa
            eval_ua += ua
            eval_mf1 += mf1
            eval_wf1 += wf1

    return (eval_loss / len(dataloader), 
            eval_wa / len(dataloader), 
            eval_ua / len(dataloader), 
            eval_mf1 / len(dataloader), 
            eval_wf1 / len(dataloader))

def train_and_evaluate(model, train_loader, val_loader, num_epochs, lr=0.0001, save_path=None):
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)

    best_val_loss = float('inf')
    best_wa, best_ua, best_mf1, best_wf1 = 0.0, 0.0, 0.0, 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_wa, train_ua, train_mf1, train_wf1 = train_step(model, train_loader, optimizer, criterion)
        val_loss, val_wa, val_ua, val_mf1, val_wf1 = eval_step(model, val_loader, criterion)

        lr_scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wa, best_ua = val_wa, val_ua
            best_mf1, best_wf1 = val_mf1, val_wf1
            
            # Save the best model
            if save_path:
                torch.save(model.state_dict(), save_path)
    
            print(f"Validation loss reduced. Best Val WA: {best_wa:.4f}, Best Val UA: {best_ua:.4f}, Best MF1: {best_mf1:.4f}, Best WF1: {best_wf1:.4f}")
            print("-" * 50)

        print(f"Train Loss: {train_loss:.4f}, Train WA: {train_wa:.4f}, Train UA: {train_ua:.4f}, Train MF1: {train_mf1:.4f}, Train WF1: {train_wf1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val WA: {val_wa:.4f}, Val UA: {val_ua:.4f}, Val MF1: {val_mf1:.4f}, Val WF1: {val_wf1:.4f}")
        print("-" * 50)

    print(f"Best Val WA: {best_wa:.4f}, Best Val UA: {best_ua:.4f}, Best Val MF1: {best_mf1:.4f}, Best Val WF1: {best_wf1:.4f}")


def calculate_accuracy(y_pred, y_true):
    wa = balanced_accuracy_score(y_true, y_pred)
    
    ua = accuracy_score(y_true, y_pred)
    
    wf1 = f1_score(y_true, y_pred, average='weighted')
    
    uf1 = f1_score(y_true, y_pred, average='macro')
    
    return wa, ua, wf1, uf1

def model_prediction(model, dataloader, metrics_fn):
    eval_wa = 0.0
    eval_ua = 0.0
    eval_wf1 = 0.0
    eval_uf1 = 0.0
    y_true_ls = []
    y_pred_ls = []
    
    model.eval()
    with torch.no_grad():
        for batch, (text_embed, audio_embed, label) in enumerate(dataloader):
            text_embed = text_embed.to(device)
            audio_embed = audio_embed.to(device)
            label = label.to(device)
            
            output_logits, output_softmax = model(text_embed, audio_embed)
            output_logits, output_softmax = output_logits.to(device), output_softmax.to(device)
            y_preds = output_softmax.argmax(dim=1).to(device)
            
            wa, ua, wf1, uf1 = metrics_fn(y_preds.cpu().numpy(), label.cpu().numpy())
            
            y_true_ls.append(label.cpu().numpy())
            y_pred_ls.append(y_preds.cpu().numpy())
            
            eval_wa += wa
            eval_ua += ua
            eval_wf1 += wf1
            eval_uf1 += uf1

        eval_wa /= len(dataloader)
        eval_ua /= len(dataloader)
        eval_wf1 /= len(dataloader)
        eval_uf1 /= len(dataloader)
        
        print(f"Total Test WA: {eval_wa:.4f} | Total Test UA: {eval_ua:.4f} | Total Test WF1: {eval_wf1:.4f} | Total Test UF1: {eval_uf1:.4f}")
        
        return eval_wa, eval_ua, eval_wf1, eval_uf1, y_true_ls, y_pred_ls
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False