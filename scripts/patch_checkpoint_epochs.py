"""
Patch last.pt checkpoint to extend epochs from 30 to 50.
YOLOv8 stores training args inside the .pt file; we need to modify 'epochs' there.
Also must reset epoch counter from -1 (done sentinel) to 29 (continue from epoch 30).
"""
import torch
import shutil
from pathlib import Path

checkpoint_path = Path("runs/detect/models/weights/pothole_det/experiment_b_merged_yolov8m2/weights/last.pt")
backup_path = checkpoint_path.with_name("last_ep30_backup.pt")

# Load from backup to avoid double-patching
source = backup_path if backup_path.exists() else checkpoint_path
print(f"Loading: {source}")
ckpt = torch.load(source, map_location="cpu", weights_only=False)

print(f"\nBefore patch:")
print(f"  epoch:        {ckpt.get('epoch', 'N/A')}")
if 'train_args' in ckpt:
    print(f"  train_args.epochs: {ckpt['train_args'].get('epochs', 'N/A')}")

# Patch 1: Set target epochs to 50
if 'train_args' in ckpt:
    ckpt['train_args']['epochs'] = 50
elif 'args' in ckpt:
    ckpt['args']['epochs'] = 50

# Patch 2: Reset epoch counter from -1 (done) to 29 (so YOLO resumes from 30)
# YOLO checks: if epoch + 1 >= epochs -> training done
# We want epoch = 29 and epochs = 50, so it continues from epoch 30
ckpt['epoch'] = 29

print(f"\nAfter patch:")
print(f"  epoch:        {ckpt['epoch']}  (was -1)")
if 'train_args' in ckpt:
    print(f"  train_args.epochs: {ckpt['train_args']['epochs']}  (was 30)")

# Save patched checkpoint
torch.save(ckpt, checkpoint_path)
print(f"\nSaved patched checkpoint: {checkpoint_path}")
print("Ready to resume with: yolo train resume model=<path>")
