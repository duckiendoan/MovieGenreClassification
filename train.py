import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall, MultilabelAccuracy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
import gc
import argparse
import datasets
import models
from utils import *
from losses import AsymmetricLossOptimized

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="seed of the experiment")
    parser.add_argument("--validation-ratio", type=float, default=0.1,
                        help="how much to split train/validation dataset")
    parser.add_argument('--shuffle', default=False, action=argparse.BooleanOptionalAction,
                        help="whether to shuffle dataset")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for dataloader")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="the learning rate of the optimizer")
    parser.add_argument("--model", type=str, default="JointModel",
                        help="The model to use from models module")
    parser.add_argument("--language-model", type=str, default="distilbert-base-uncased",
                        help="Language model used for movie title processing from HuggingFace transformers")
    parser.add_argument('--track', default=False, action=argparse.BooleanOptionalAction,
                        help="if toggled, this run will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ML_MovieLens",
                        help="the wandb's project name")

    args, unknowns = parser.parse_known_args()
    # fmt: on
    return args


def data_loader(movies_train, movies_test, genres, tokenizer, batch_size=8, validation_ratio=0.1, shuffle=True):
    def crop_square(img):
        return transforms.functional.crop(img, 0, 0, img.width, img.width)
    
    mean = [0.50248874, 0.43833769, 0.412207]
    std = [0.36760546, 0.35544249, 0.34853585]
    normalize = transforms.Normalize(mean=mean, std=std)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(crop_square),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    train_set = datasets.MovieLensDataset(movies_train, genres, tokenizer=tokenizer, transform=transform)
    valid_set = datasets.MovieLensDataset(movies_train, genres, tokenizer=tokenizer, transform=transform)
    test_set = datasets.MovieLensDataset(movies_test, genres, tokenizer=tokenizer, transform=transform)

    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(validation_ratio * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader, test_loader

def evaluate(model, dataloader, metric, device):
    model.eval()
    
    for input_ids, attention_mask, img_tensor, label in dataloader:
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            img_tensor = img_tensor.to(device)
            label = label.to(device)

            out = model(input_ids, attention_mask, img_tensor)
            metric.update(out, label)
    return metric.compute()

if __name__ == "__main__":
    if not os.path.exists("./ml1m"):
        datasets.download()
    args = parse_args()
    print(vars(args))
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load data
    movies_train, movies_test = datasets.load_data_frames()
    genres = datasets.load_genres()
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    train_loader, valid_loader, test_loader = data_loader(movies_train, movies_test, genres,
                                                          tokenizer, batch_size=args.batch_size, 
                                                          validation_ratio=args.validation_ratio,
                                                          shuffle=args.shuffle)

    # Pretrained models
    bert = AutoModel.from_pretrained(args.language_model)
    resnet50 = torchvision.models.resnet50(progress=True, weights=torchvision.models.ResNet50_Weights.DEFAULT)

    # Model
    model_class = getattr(models, args.model)
    print(f"Using {model_class.__name__}")
    model = model_class(resnet50, bert, num_classes=len(genres)).to(device)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = AsymmetricLossOptimized()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5,
                                                           patience=3, min_lr=1e-6, verbose=True)
    # Metrics
    train_metrics = MetricCollection([
        MultilabelF1Score(num_labels=len(genres), threshold=0.5, average='macro'),
        MultilabelPrecision(num_labels=len(genres), threshold=0.5, average='macro'),
        MultilabelRecall(num_labels=len(genres), threshold=0.5, average='macro'),
        MultilabelAccuracy(num_labels=len(genres), threshold=0.5, average='macro')
    ]).to(device)

    valid_metrics = MetricCollection([
        MultilabelF1Score(num_labels=len(genres), threshold=0.5, average='macro'),
        MultilabelPrecision(num_labels=len(genres), threshold=0.5, average='macro'),
        MultilabelRecall(num_labels=len(genres), threshold=0.5, average='macro'),
        MultilabelAccuracy(num_labels=len(genres), threshold=0.5, average='macro')
    ]).to(device)

    test_metrics = MetricCollection([
        MultilabelF1Score(num_labels=len(genres), threshold=0.5, average='macro'),
        MultilabelPrecision(num_labels=len(genres), threshold=0.5, average='macro'),
        MultilabelRecall(num_labels=len(genres), threshold=0.5, average='macro'),
        MultilabelAccuracy(num_labels=len(genres), threshold=0.5, average='macro')
    ]).to(device)

    if args.track:
        import wandb
        wandb.login()
        wandb.init(project="ML_MovieLens",
                   config=vars(args),
                   sync_tensorboard=True,
                   save_code=True)
    
    writer = SummaryWriter(f"runs/ResNetBert_{datetime.now().strftime('%d-%m-%Y %H-%M-%S')}")
    writer.add_text("model", str(model))
    global_step = 0
    # Train
    for e in range(args.epochs):
        print(f"Epoch #{e}")
        model.train()
        for i, (input_ids, attention_mask, img_tensor, label) in enumerate(train_loader):
            global_step += 1
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            img_tensor = img_tensor.to(device)
            label = label.to(device)

            out = model(input_ids, attention_mask, img_tensor)
            loss = loss_fn(out, label)
            metrics = train_metrics(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("losses/train_loss", loss.item(), global_step)

        print(f"\tTrain loss: {loss.item()}")
        model.eval()
        for input_ids, attention_mask, img_tensor, label in valid_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            img_tensor = img_tensor.to(device)
            label = label.to(device)

            with torch.no_grad():
                out = model(input_ids, attention_mask, img_tensor)
                loss = loss_fn(out, label)
                valid_metrics.update(out, label)

        writer.add_scalar("losses/valid_loss", loss.item(), e)
        total_train_metrics = train_metrics.compute()
        total_valid_metrics = valid_metrics.compute()
        scheduler.step(total_valid_metrics['MultilabelF1Score'].cpu().item())
        for k, v in total_train_metrics.items():
            metric_name = k.replace('Multilabel', '').lower()
            writer.add_scalar(f"metrics/train_{metric_name}", v.item(), e)
            writer.add_scalar(f"metrics/validation_{metric_name}", total_valid_metrics[k].item(), e)

        print(f"\tValidation loss: {loss.item()}")
        print(f"\tTrain: {pretty_metrics(total_train_metrics)}")
        print(f"\tValidation: {pretty_metrics(total_valid_metrics)}")
        print()

        del input_ids, attention_mask, img_tensor
        gc.collect()
        torch.cuda.empty_cache()
        train_metrics.reset()
        valid_metrics.reset()

    test_result = evaluate(model, test_loader, test_metrics, device)
    writer.add_text("test_result", str(pretty_metrics(test_result)))
    print("Test result:")
    print(pretty_metrics(test_result))
    writer.close()
    if args.track:
        wandb.finish()