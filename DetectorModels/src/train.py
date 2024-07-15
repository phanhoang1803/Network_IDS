"""
Script to train the model
"""

from collections import defaultdict
import gc
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm
import copy
from data.data_loading import load_data
from data.dataset import UNSW_NB15_Dataset
from model.criterion import criterion
from utils.utils import fetch_model, parse_args, set_seed, make_dir

def train_epoch(model, dataloader, optimizer, scheduler, epoch, device, CONFIG):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): The data loader for the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler for the optimizer. Defaults to None.
        epoch (int): The current epoch number.
        device (torch.device): The device to run the model on.
        CONFIG (dict): The configuration dictionary.

    Returns:
        tuple: A tuple containing the epoch loss and epoch accuracy.
    """
    model.train()
    
    # Initialize the variables to keep track of the running loss and correct predictions
    running_loss = 0.0
    running_correct = 0
    
    # Create a tqdm bar to display the progress
    train_bar = tqdm(dataloader, total=len(dataloader))
    for data in train_bar:
        x = data["x"].to(device, dtype=torch.float, non_blocking=True)
        y = data["y"].to(device, dtype=torch.long, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(x)
        
        loss = criterion(outputs, y)
        loss.backward()
        
        # Update the weights
        optimizer.step()

        if scheduler is not None:
            # Update the learning rate
            scheduler.step()

        # Update the running loss and correct predictions
        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == y).sum().item()
        
        train_bar.set_postfix(Loss=running_loss / len(dataloader.dataset), Accuracy=running_correct / len(dataloader.dataset))
        
    # Clean up memory
    gc.collect()
    
    return running_loss / len(dataloader.dataset), running_correct / len(dataloader.dataset)

def valid_epoch(model, dataloader, epoch, device, CONFIG):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The data loader for the validation dataset.
        epoch (int): The current epoch number.
        device (torch.device): The device to run the model on.
        CONFIG (dict): The configuration dictionary.

    Returns:
        tuple: A tuple containing the epoch loss and epoch accuracy.
    """
    
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize the variables to keep track of the running loss and correct predictions
    running_loss = 0.0
    running_correct = 0.0
    
    # Create a tqdm bar to display the progress
    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            # Move the data to the device
            x = data["x"].to(device, dtype=torch.float)
            y = data["y"].to(device, dtype=torch.long)
            
            # Forward pass
            outputs = model(x)
            
            # Calculate the loss
            loss = criterion(outputs, y)

            # Update the running loss and correct predictions
            running_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == y).item()
            
    # Calculate the epoch loss and accuracy
    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_correct / dataset_size
    
    return epoch_loss, epoch_acc

def train(model, train_loader, valid_loader, optimizer, scheduler, device, CONFIG):
    """
    This function trains the model using the training and validation data loaders.
    
    Args:
        model: The neural network model to be trained
        train_loader: DataLoader for the training dataset
        valid_loader: DataLoader for the validation dataset
        optimizer: Optimizer for updating model parameters
        scheduler: Learning rate scheduler
        device: Device to run the training on
        CONFIG: Dictionary containing configuration parameters
    
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    
    # Initialize variables and history
    start = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_epoch_acc = -np.inf
    history = defaultdict(list)
    patience = CONFIG["patience"]
    
    current_patience = 0
    
    print("[INFO] Training started...")
    print("[INFO] Traning on {} device...".format(device))
    
    make_dir(CONFIG["save_dir"])
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        gc.collect()
        
        # Training phase
        train_epoch_loss, train_epoch_acc = train_epoch(model=model,
                                                        dataloader=train_loader,
                                                        optimizer=optimizer,
                                                        scheduler=scheduler,
                                                        epoch=epoch,
                                                        device=device,
                                                        CONFIG=CONFIG)

        # Validation phase
        valid_epoch_loss, valid_epoch_acc = valid_epoch(model=model,
                                                        dataloader=valid_loader,
                                                        epoch=epoch,
                                                        device=device,
                                                        CONFIG=CONFIG)
        
        # Update history
        history["Train Loss"].append(train_epoch_loss)
        history["Train Acc"].append(train_epoch_acc)
        history["Valid Loss"].append(valid_epoch_loss)
        history["Valid Acc"].append(valid_epoch_acc)
        history["Learning Rate"].append(scheduler.get_last_lr()[0])
        
        # Check if validation accuracy improved
        if best_epoch_acc < valid_epoch_acc:
            print(f"[INFO] Validation accuracy increased from {best_epoch_acc:.4f} --> {valid_epoch_acc:.4f}. Saving model...")
            best_epoch_acc = valid_epoch_acc
            current_patience = 0
            best_model = copy.deepcopy(model.state_dict())
            
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.bin"))
        else:
            current_patience += 1
            if current_patience >= patience:
                print("[INFO] Validation accuracy did not increase for {} epochs. Stopping training...".format(patience))
                break
            
        # Print epoch summary
        print(f"[INFO] Epoch {epoch} | "
              f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.4f} | "
              f"Valid Loss: {valid_epoch_loss:.4f} | Valid Acc: {valid_epoch_acc:.4f}\n")
        
    end = time.time()
    time_elapsed = end - start
    print(f"[INFO] Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"[INFO] Best valid accuracy: {best_epoch_acc:.4f}")
    print("[INFO] Training completed.")
    
    # Load the best model
    model.load_state_dict(best_model)
    
    return model, history
    
def test(model, test_loader, device, CONFIG):
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            # Move the data to the device
            x = data["x"].to(device, dtype=torch.float)
            y = data["y"].to(device, dtype=torch.long)
            
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    
    # Print or log metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    
    return accuracy, f1, recall, precision

def fetch_scheduler(optimizer, CONFIG):
    """
    Fetches a scheduler based on the configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to use with the scheduler.
        CONFIG (dict): The configuration dictionary containing the scheduler type and
            its parameters.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: The scheduler if the scheduler type
            is valid, None otherwise.
    """
    
    # CosineAnnealingLR scheduler
    if CONFIG["scheduler"] == "CosineAnnealingLR":
        # CosineAnnealingLR scheduler with T_max and eta_min
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG["T_max"],
            eta_min=CONFIG["min_lr"]
        )
    
    # CosineAnnealingWarmRestarts scheduler
    elif CONFIG["scheduler"] == "CosineAnnealingWarmRestarts":
        # CosineAnnealingWarmRestarts scheduler with T_0 and eta_min
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=CONFIG["T_0"],
            eta_min=CONFIG["min_lr"]
        )
    
    else:
        return None
    
    return scheduler

def fetch_optimizer(model, CONFIG):
    """
    Fetches an optimizer based on the configuration.

    Args:
        model (nn.Module): The model to optimize.
        CONFIG (dict): The configuration dictionary containing the optimizer type and
            its parameters.

    Returns:
        torch.optim.Optimizer or None: The optimizer if the optimizer type is valid, None otherwise.
    """
    # Adam optimizer
    if CONFIG["optimizer"] == "Adam":
        # Adam optimizer with learning rate and weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    
    # SGD optimizer
    elif CONFIG["optimizer"] == "SGD":
        # SGD optimizer with learning rate and weight decay
        optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    
    else:
        return None
    
    return optimizer


def prepare_loaders(df, fold, CONFIG):
    """
    Prepare the data loaders for training and validation.

    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        fold (int): The fold number.
        CONFIG (dict): The configuration dictionary.

    Returns:
        tuple: A tuple containing the train loader and validation loader.
    """
    # Fill Na values with inferred data types
    df.infer_objects(copy=False)
    
    # Split the data into training and validation sets
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # Create datasets for training and validation
    train_dataset = UNSW_NB15_Dataset(df_train, CONFIG)
    valid_dataset = UNSW_NB15_Dataset(df_valid, CONFIG)
    
    # Create data loaders for training and validation
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CONFIG["train_batch_size"],
        num_workers=CONFIG["num_workers"],
        shuffle=True,
        drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=CONFIG["valid_batch_size"],
        num_workers=CONFIG["num_workers"],
        shuffle=False
    )
    
    return train_loader, valid_loader

def main():
    args, CONFIG = parse_args()
    
    set_seed(CONFIG["seed"])
    
    # Load data, preprocess data, feature engineering, ...
    train_csv = os.path.join(CONFIG["data_dir"], "UNSW_NB15_training-set.csv")
    df = load_data(train_csv, CONFIG)
    
    # Set T-max
    CONFIG['T_max'] = df.shape[0] * (CONFIG["n_fold"]-1) * CONFIG['epochs'] // CONFIG['train_batch_size'] // CONFIG["n_fold"]
    
    # Create folds
    # gkf = GroupKFold(n_splits=CONFIG["n_fold"])
    # sgkf = StratifiedGroupKFold(n_splits=CONFIG["n_fold"])
    # for fold, (_, val) in enumerate(sgkf.split(X=df, y=df["label"], groups=None)):
    #     df.loc[val, "kfold"] = fold
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=CONFIG["seed"], stratify=df["label"])
    train_dataset = UNSW_NB15_Dataset(df_train, CONFIG)
    valid_dataset = UNSW_NB15_Dataset(df_valid, CONFIG)
    
    # Create data loaders for training and validation
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CONFIG["train_batch_size"],
        num_workers=CONFIG["num_workers"],
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=CONFIG["valid_batch_size"],
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        shuffle=False
    )
    
    print(f"[INFO] Training on {len(df_train)} samples and validating on {len(df_valid)} samples.")
    
    # Get dataloaders
    # train_loader, valid_loader = prepare_loaders(df, fold=args.fold, CONFIG=CONFIG)
    CONFIG["input_dim"] = df.shape[1] - 1
    
    # Initialize model
    model = fetch_model(CONFIG)
    model.to(CONFIG["device"])
    
    # Get optimizer and scheduler
    optimizer = fetch_optimizer(model, CONFIG)
    scheduler = fetch_scheduler(optimizer, CONFIG)
    
    # Train the model
    model, history = train(model=model,
                           train_loader=train_loader,
                           valid_loader=valid_loader,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           device=CONFIG["device"],
                           CONFIG=CONFIG)
    
    history = pd.DataFrame.from_dict(history)
    history.to_csv(os.path.join(CONFIG["save_dir"], "history.csv"), index=False)
    
    # Test
    test_csv = os.path.join(CONFIG["data_dir"], "UNSW_NB15_testing-set.csv")
    df_test = load_data(test_csv, CONFIG)
    test_dataset = UNSW_NB15_Dataset(df_test, CONFIG)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CONFIG["valid_batch_size"],
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        shuffle=False
    )
    
    test(model, test_loader, CONFIG["device"], CONFIG)
    
if __name__ == "__main__":
    main()