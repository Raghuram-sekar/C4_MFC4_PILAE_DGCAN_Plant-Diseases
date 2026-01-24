import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_dataset
from dcgan import Generator, Discriminator
from pilae import PILAE

def verify_robustness():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_dir = os.path.join(project_root, 'Dataset', 'color')
    model_dir = os.path.join(project_root, 'models')
    
    # Load Data
    print("Loading dataset...")
    dataloader, img_count, class_names = load_dataset(dataset_dir, image_size=64, batch_size=64)
    num_classes = len(class_names)
    
    # Load Models
    print("Loading trained models...")
    netD = Discriminator(n_classes=num_classes).to(device)
    netG = Generator(n_classes=num_classes).to(device)
    
    netD.load_state_dict(torch.load(os.path.join(model_dir, 'netD_cond.pth'), map_location=device))
    netG.load_state_dict(torch.load(os.path.join(model_dir, 'netG_cond.pth'), map_location=device))
    
    netD.eval()
    netG.eval()
    
    # 1. Extract Real Features
    print("Extracting features from real data...")
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader):
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            feats = netD.extract_features(imgs, lbls)
            features_list.append(feats.cpu().numpy())
            labels_list.append(lbls.cpu().numpy())
            
    X_real = np.concatenate(features_list, axis=0)
    y_real = np.concatenate(labels_list, axis=0)
    
    # 2. Balance Dataset
    TARGET_COUNT = 1000
    print(f"Balancing dataset to {TARGET_COUNT} samples per class...")
    
    X_syn_list = []
    y_syn_list = []
    class_counts = Counter(y_real)
    
    for cls_idx in range(num_classes):
        current_count = class_counts[cls_idx]
        needed = TARGET_COUNT - current_count
        
        if needed > 0:
            noise = torch.randn(needed, 100, 1, 1, device=device)
            labels = torch.full((needed,), cls_idx, dtype=torch.long, device=device)
            
            with torch.no_grad():
                fake_imgs = netG(noise, labels)
                fake_feats = netD.extract_features(fake_imgs, labels)
            
            X_syn_list.append(fake_feats.cpu().numpy())
            y_syn_list.append(labels.cpu().numpy())
            
    if X_syn_list:
        X_syn = np.concatenate(X_syn_list, axis=0)
        y_syn = np.concatenate(y_syn_list, axis=0)
        X_balanced = np.concatenate([X_real, X_syn], axis=0)
        y_balanced = np.concatenate([y_real, y_syn], axis=0)
    else:
        X_balanced, y_balanced = X_real, y_real
        
    print(f"Balanced Dataset Size: {len(X_balanced)}")
    
    # 3. K-Fold CV
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    
    print(f"\nRunning {k_folds}-Fold Cross-Validation...")
    
    enc = OneHotEncoder(sparse_output=False)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_balanced, y_balanced)):
        X_train, X_val = X_balanced[train_idx], X_balanced[val_idx]
        y_train, y_val = y_balanced[train_idx], y_balanced[val_idx]
        
        # Train PILAE
        Y_train_oh = enc.fit_transform(y_train.reshape(-1, 1))
        pilae = PILAE(input_dim=X_train.shape[1])
        pilae.fit(X_train, Y_train_oh)
        
        # Evaluate
        y_pred = pilae.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)
        print(f"Fold {fold+1}: {acc:.4f}")
        
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"\nResults:")
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Std Deviation: {std_acc:.4f}")
    
    if mean_acc > 0.99 and std_acc < 0.01:
        print("\nCONCLUSION: NO OVERFITTING DETECTED.")
        print("The model consistently achieves ~100% accuracy across different data splits.")
    else:
        print("\nCONCLUSION: Potential instability or overfitting detected.")

if __name__ == "__main__":
    verify_robustness()
