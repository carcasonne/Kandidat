import os
from collections import Counter
from datetime import datetime
from os.path import split

import timm
from mpl_toolkits.mplot3d.proj3d import transform
from torch.utils.data import random_split
import numpy as np
import librosa
import soundfile
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, ASTForAudioClassification, ASTConfig, ASTModel
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
import wandb
import plotly.graph_objects as go
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torchvision import datasets, transforms

from Datasets import *
from attention_map import generate_enhanced_attention_maps
from wandb_login import login
from benchmark import benchmark

samples = {"bonafide": 100000, "fake":100000}
EPOCHS = 1
TRAIN_TEST_SPLIT = 0.2
layers_to_freeze = 10

# Define dataset path
ADD_DATASET_PATH = r"spectrograms/ADD"
FOR_DATASET_PATH = r"spectrograms/FoR/for-2sec/for-2seconds"
FOR_DATASET_PATH_TRAINING = r"spectrograms/FoR/for-2sec/for-2seconds/Training"
FOR_DATASET_PATH_TESTING = r"spectrograms/FoR/for-2sec/for-2seconds/Testing"
ASVS_DATASET_PATH = r"spectrograms"

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_pretrain_model(device='cuda'):
    # Load Pretrained ViT Model
    model_name = "vit_base_patch16_224"
    model = timm.create_model(model_name, pretrained=True, in_chans=1)

    # Modify classifier head for binary classification
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 2)  # Binary classification (2 classes)

    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    # Unfreeze the final classification layer (classifier head)
    for param in model.head.parameters():
        param.requires_grad = True  # Unfreeze the classifier

    # Unfreeze the layer before the head (typically the final transformer block)
    # In ViT, this is usually the last block in model.blocks
    for param in model.blocks[-1].parameters():
        param.requires_grad = True
    model.to(device)

    return model

def setup_ast_model(model_name, embedding_size, frozen_layers, device='cuda'):
    # Your desired model name and input length
    model = ASTForAudioClassification.from_pretrained(model_name)
    model.config.max_length = embedding_size
    model.config.num_labels = 2
    model.config.id2label = {0: "bonafide", 1: "spoof"}
    model.config.label2id = {"bonafide": 0, "spoof": 1}

    # Modify the classifier for 2 classes
    model.classifier.dense = nn.Linear(model.classifier.dense.in_features, 2)  # Change output layer
    model.classifier.out_proj = nn.Linear(2, 2)  # Adjust projection layer

    # Interpolate positional embeddings
    desired_max_length = 0
    if embedding_size == 200:
        desired_max_length = 230
    elif embedding_size == 450:
        desired_max_length = 530

    position_embeddings = model.audio_spectrogram_transformer.embeddings.position_embeddings  # shape: (1, old_len, dim)
    old_len = position_embeddings.shape[1]
    if old_len != desired_max_length:
        print(f"Interpolating position embeddings from {old_len} to {desired_max_length}")
        # Use interpolation
        interpolated_pos_emb = F.interpolate(
            position_embeddings.permute(0, 2, 1),  # shape: (1, dim, old_len)
            size=desired_max_length,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1)  # shape back to (1, new_len, dim)

        model.audio_spectrogram_transformer.embeddings.position_embeddings = nn.Parameter(interpolated_pos_emb)

    print("Final position embedding shape:", model.audio_spectrogram_transformer.embeddings.position_embeddings.shape)
    model.audio_spectrogram_transformer.embeddings.position_embeddings.requires_grad = False

    for i in range(frozen_layers):  # Layers 0 to 9
        for param in model.audio_spectrogram_transformer.encoder.layer[i].parameters():
            param.requires_grad = False

    model.to(device)
    return model



def train_ast(model, train_loader, val_loader, criterion, optimizer, num_epochs, flavor_text, seed):
    login()
    wandb.init(project="Kandidat-AST", entity="Holdet_thesis")
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        true_labels, pred_labels = [], []  # Lists to store predictions and true labels

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]")
        for batch in loop:
            inputs, labels = batch["input_values"].to(device), batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_values=inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Append true labels and predictions to the lists for confusion matrix
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        # Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        tn, fp, fn, tp = cm.ravel()

        # Compute Metrics
        loss = running_loss / len(train_loader)
        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * tp) / ((2 * tp) + fp + fn)

        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        values = [acc / 100, precision, recall, f1]  # Normalize accuracy to [0,1]
        values.append(values[0])  # Close the radar chart

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],  # Close the circle
            fill='toself',
            name=f'Epoch {epoch + 1}'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)

        wandb.log({"train_conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                                 y_true=true_labels, preds=pred_labels,
                                                                 class_names=["Real", "Fake"])})

        # Log to Weights & Biases
        wandb.log({
            "Train Accuracy": acc,
            "Train Loss": loss,
            "Train Precision": precision,
            "Train Recall": recall,
            "Train F1 Score": f1,
            "Train Spider Plot": fig,
            "Seed": seed
        })

        # --------- VALIDATION LOOP ---------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_true_labels, val_pred_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]"):
                inputs, labels = batch["input_values"].to(device), batch["labels"].to(device)

                outputs = model(input_values=inputs).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_true_labels.extend(labels.cpu().numpy())
                val_pred_labels.extend(preds.cpu().numpy())

        val_cm = confusion_matrix(val_true_labels, val_pred_labels)
        tn, fp, fn, tp = val_cm.ravel()
        val_loss /= len(val_loader)
        val_acc = (tp + tn) / (tp + tn + fp + fn)
        val_precision = tp / (tp + fp)
        val_recall = tp / (tp + fn)
        val_f1 = (2 * tp) / ((2 * tp) + fp + fn)
        val_values = [val_acc / 100, val_precision, val_recall, val_f1]
        val_values.append(val_values[0])

        val_fig = go.Figure()
        val_fig.add_trace(go.Scatterpolar(
            r=val_values,
            theta=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            fill='toself',
            name=f'Val Epoch {epoch + 1}'
        ))
        val_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        wandb.log({"val_conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                               y_true=val_true_labels, preds=val_pred_labels,
                                                               class_names=["Real", "Fake"])})
        wandb.log({
            "Val Accuracy": val_acc,
            "Val Loss": val_loss,
            "Val Precision": val_precision,
            "Val Recall": val_recall,
            "Val F1 Score": val_f1,
            "Val Spider Plot": val_fig
        })

        if (epoch % 5 == 0 and epoch != 0) or epoch == EPOCHS - 1:
            save_dir = "checkpoints"
            os.makedirs(save_dir, exist_ok=True)

            date = datetime.now().strftime("%Y%m%d_%H%M%S")
            save = os.path.join(save_dir, f"asvspoof-ast-model_{flavor_text}_{epoch}_{date}")
            model.save_pretrained(save)

        print(f"Epoch {epoch + 1}: Train Loss = {loss:.4f}, Train Acc = {acc:.2f}%, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%, "
              f"Val Precision = {val_precision:.4f}, Val Recall = {val_recall:.4f}, Val F1 = {val_f1:.4f}")

    wandb.finish()
    return model

def train_pretrain(model, train_loader, val_loader, criterion, optimizer, num_epochs, flavor_text, seed):
    # Training Loop
    login()
    wandb.init(project="Kandidat-Pre-trained", entity="Holdet_thesis")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        true_labels, pred_labels = [], []  # Lists to store predictions and true labels
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Append true labels and predictions to the lists for confusion matrix
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        # Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        tn, fp, fn, tp = cm.ravel()

        # Compute Metrics
        loss = running_loss / len(train_loader)
        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * tp) / ((2 * tp) + fp + fn)

        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        values = [acc / 100, precision, recall, f1]  # Normalize accuracy to [0,1]
        values.append(values[0])  # Close the radar chart

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],  # Close the circle
            fill='toself',
            name=f'Epoch {epoch + 1}'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)

        wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                           y_true=true_labels, preds=pred_labels,
                                                           class_names=["Real", "Fake"])})

        # Log to Weights & Biases
        wandb.log({
            "Accuracy": acc,
            "Loss": loss,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Spider Plot": fig,
            "Seed": seed
        })

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        val_true_labels, val_pred_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Validation]"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_true_labels.extend(labels.cpu().numpy())
                val_pred_labels.extend(preds.cpu().numpy())

        val_cm = confusion_matrix(val_true_labels, val_pred_labels)
        tn, fp, fn, tp = val_cm.ravel()
        val_loss /= len(val_loader)
        val_acc = (tp + tn) / (tp + tn + fp + fn)
        val_precision = tp / (tp + fp)
        val_recall = tp / (tp + fn)
        val_f1 = (2 * tp) / ((2 * tp) + fp + fn)

        val_values = [val_acc / 100, val_precision, val_recall, val_f1]
        val_values.append(val_values[0])

        val_fig = go.Figure()
        val_fig.add_trace(go.Scatterpolar(
            r=val_values,
            theta=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            fill='toself',
            name=f'Val Epoch {epoch + 1}'
        ))
        val_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)

        wandb.log({
            "Val Confusion Matrix": wandb.plot.confusion_matrix(probs=None, y_true=val_true_labels,
                                                                preds=val_pred_labels,
                                                                class_names=["Real", "Fake"]),
            "Val Accuracy": val_acc,
            "Val Loss": val_loss,
            "Val Precision": val_precision,
            "Val Recall": val_recall,
            "Val F1 Score": val_f1,
            "Val Spider Plot": val_fig
        })
        if (epoch % 5 == 0 and epoch != 0) or epoch == num_epochs - 1:
            save_dir = "checkpoints"
            os.makedirs(save_dir, exist_ok=True)

            date = datetime.now().strftime("%Y%m%d_%H%M%S")
            save = os.path.join(save_dir, f"asvspoof-pretrain-model_{flavor_text}_{epoch}_{date}")

            torch.save(model.state_dict(), save)

        print(f"Epoch {epoch + 1}: Train Loss = {loss:.4f}, Train Acc = {acc:.2f}%, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%, "
              f"Val Precision = {val_precision:.4f}, Val Recall = {val_recall:.4f}, Val F1 = {val_f1:.4f}")
    wandb.finish()
    return model

def ast_train_ADD_bench_attention():
    embedding_size = 450
    model = setup_ast_model(MODEL_NAME, embedding_size, layers_to_freeze)
    print(f"Model setup complete")

    train_load, val_load, seed = load_ADD_dataset(ADD_DATASET_PATH, samples, True, TRAIN_TEST_SPLIT, embedding_size=embedding_size)
    cri = nn.CrossEntropyLoss()
    opti = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    print(f"Starting to train")
    flavor_text = "ADD_data"
    trained_model = train_ast(model, train_load, val_load, cri, opti, EPOCHS, flavor_text, seed)

    print(f"Model completed training")
    print(f"Benchmark AST trained on ADD, on ASV")
    asv_data, _, _ = load_ASV_dataset(ASVS_DATASET_PATH, samples, is_AST=True, split=None, transform=None, embedding_size=embedding_size)
    #benchmark(trained_model, asv_data, flavor_text="Benchmark AST trained on ADD, on ASV", is_AST=True)

    print(f"Benchmark AST Trained on ADD, on FoR")
    for_data = load_FOR_total(FOR_DATASET_PATH, samples, is_AST=True, embedding_size=embedding_size)
    #benchmark(trained_model, for_data, flavor_text="Benchmark AST Trained on ADD, on FoR", is_AST=True)

    print(f"Generating Attention_maps")
    generate_enhanced_attention_maps(trained_model ,asv_data, num_samples=10, flavor_text="AST_trn_ADD_on_ASV")
    generate_enhanced_attention_maps(trained_model ,train_load, num_samples=10, flavor_text="AST_trn_ADD_on_ADD")
    generate_enhanced_attention_maps(trained_model ,for_data, num_samples=10, flavor_text="AST_trn_ADD_on_FoR")


def ast_train_FoR_bench_attention(vson: bool):
    if vson:
        ADD_DATASET_PATH = r"/home/alsk/Kanidat/AST/spectrograms/ADD"
        FOR_DATASET_PATH = r"/home/alsk/Kanidat/AST/spectrograms/FoR/for-2sec/for-2seconds"
        FOR_DATASET_PATH_TRAINING = r"/home/alsk/Kanidat/AST/spectrograms/FoR/for-2sec/for-2seconds/Training"
        FOR_DATASET_PATH_TESTING = r"/home/alsk/Kanidat/AST/spectrograms/FoR/for-2sec/for-2seconds/Testing"
        ASVS_DATASET_PATH = r"/home/alsk/Kanidat/AST/spectrograms"

    embedding_size = 200
    model = setup_ast_model(MODEL_NAME, embedding_size, layers_to_freeze)
    print(f"Model setup complete")

    train_load, val_load, seed = load_FOR_dataset(FOR_DATASET_PATH_TRAINING, FOR_DATASET_PATH_TESTING, True, samples, None, embedding_size)
    cri = nn.CrossEntropyLoss()
    opti = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    print(f"Starting to train")
    flavor_text = "ADD_data"
    trained_model = train_ast(model, train_load, val_load, cri, opti, EPOCHS, flavor_text, seed)

    print(f"Model completed training")
    print(f"Benchmark AST trained on FoR, on ASV")
    asv_data, _, _ = load_ASV_dataset(ASVS_DATASET_PATH, samples, True, split=None, embedding_size=embedding_size)
    benchmark(trained_model, asv_data, flavor_text="Benchmark AST trained on FoR, on ASV", is_AST=True)

    print(f"Benchmark AST Trained on FoR, on ADD")
    add_data, _, _ = load_ADD_dataset(ADD_DATASET_PATH, samples, is_AST=True, split=None, embedding_size=embedding_size)
    benchmark(trained_model, add_data, flavor_text="Benchmark AST Trained on ADD, on FoR", is_AST=True)

    print(f"Generating Attention_maps")
    generate_enhanced_attention_maps(trained_model ,asv_data, num_samples=10, flavor_text="AST_trn_FoR_on_ASV")
    generate_enhanced_attention_maps(trained_model ,train_load, num_samples=10, flavor_text="AST_trn_FoR_on_FoR")
    generate_enhanced_attention_maps(trained_model ,add_data, num_samples=10, flavor_text="AST_trn_FoR_on_ADD")

def pre_train_ADD_bench_attention():
    model = setup_pretrain_model()
    print(f"Model setup complete")

    # Define Data Transforms
    transform = transforms.Compose([
        StretchMelCropTime(224, 224),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
    train_load, val_load, seed = load_ADD_dataset(ADD_DATASET_PATH, samples, False, TRAIN_TEST_SPLIT, transform)
    cri = nn.CrossEntropyLoss()
    opti = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    print(f"Starting to train")
    flavor_text = "ADD_data"
    trained_model = train_pretrain(model, train_load, val_load, cri, opti, EPOCHS, flavor_text, seed)

    print(f"Model completed training")
    print(f"Benchmark Pre-trained VIT trained on ADD, on ASV")
    asv_data, _, _ = load_ASV_dataset(ASVS_DATASET_PATH, samples, False, split=None, transform=transform)
    benchmark(trained_model, asv_data, flavor_text="Benchmark Pre-trained VIT trained on ADD, on ASV", is_AST=False)

    print(f"Benchmark Pre-trained VIT trained on ADD, on FoR")
    for_data = load_FOR_total(FOR_DATASET_PATH, samples, False, transform)
    benchmark(trained_model, for_data, flavor_text="Benchmark Pre-trained VIT trained on ADD, on FoR", is_AST=False)

    # doest work with pretrained vit
    #print(f"Generating Attention_maps")
    #generate_enhanced_attention_maps(trained_model, asv_data, num_samples=10)
    #generate_enhanced_attention_maps(trained_model, train_load, num_samples=10)
    #generate_enhanced_attention_maps(trained_model, for_data, num_samples=10)


def pre_train_FoR_bench_attention():
    model = setup_pretrain_model()
    print(f"Model setup complete")

    # Define Data Transforms
    transform = transforms.Compose([
        StretchMelCropTime(224, 224),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
    train_load, val_load, seed = load_FOR_dataset(FOR_DATASET_PATH_TRAINING, FOR_DATASET_PATH_TESTING, False, samples, transform)
    cri = nn.CrossEntropyLoss()
    opti = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    print(f"Starting to train")
    flavor_text = "FoR_data"
    trained_model = train_pretrain(model, train_load, val_load, cri, opti, EPOCHS, flavor_text, seed)

    print(f"Model completed training")
    print(f"Benchmark Pre-trained VIT trained on FoR, on ASV")
    asv_data, _, _ = load_ASV_dataset(ASVS_DATASET_PATH, samples, False, split=None, transform=transform)
    benchmark(trained_model, asv_data, flavor_text="Benchmark Pre-trained VIT trained on FoR, on ASV", is_AST=False)

    print(f"Benchmark Pre-trained VIT trained on FoR, on ADD")
    for_data, _, _ = load_ADD_dataset(ADD_DATASET_PATH, samples, False, split=None, transform=transform)
    benchmark(trained_model, for_data, flavor_text="Benchmark Pre-trained VIT trained on FoR, on ADD", is_AST=False)

    # doesnt work with pretrained vit
    #print(f"Generating Attention_maps")
    #generate_enhanced_attention_maps(trained_model, asv_data, num_samples=10)
    #generate_enhanced_attention_maps(trained_model, train_load, num_samples=10)
    #generate_enhanced_attention_maps(trained_model, for_data, num_samples=10)
