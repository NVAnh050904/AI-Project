import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.upar.loader import UPARDataset, get_upar_transforms

def main():
    print("=" * 60)
    print("TEST: TESTING MULTI-HEAD UPAR DATASET LOADER")
    print("=" * 60)
    
    train_transform, val_transform = get_upar_transforms(height=256, width=128)
    
    dataset = UPARDataset(split='all', transform=val_transform)
    ds_len = len(dataset)
    print(f"Dataset length: {ds_len}")
    assert ds_len == 145656, f"Expected dataset length 145656, got {ds_len}"
    
    img, raw_label, head_targets, ds_id, img_name = dataset[0]
    print(f"Sample image name: {img_name}")
    print(f"Sample dataset ID: {ds_id}")
    print(f"Single image shape: {img.shape}")
    print(f"Single raw label shape: {raw_label.shape}")
    print(f"Head targets keys: {list(head_targets.keys())}")
    
    assert img.shape == torch.Size([3, 256, 128]), f"Expected image shape [3, 256, 128], got {img.shape}"
    assert raw_label.shape == torch.Size([40]), f"Expected label shape [40], got {raw_label.shape}"
    assert len(head_targets) == 11, f"Expected 11 head targets, got {len(head_targets)}"
    assert head_targets["age"].dtype == torch.long, "Age target must be torch.long for CrossEntropy"
    
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    batch_imgs, batch_labels, batch_head_targets, batch_ids, batch_names = next(iter(loader))
    
    print(f"Batch images shape: {batch_imgs.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    
    assert batch_imgs.shape == torch.Size([16, 3, 256, 128]), f"Expected batch imgs shape [16, 3, 256, 128], got {batch_imgs.shape}"
    assert batch_labels.shape == torch.Size([16, 40]), f"Expected batch labels shape [16, 40], got {batch_labels.shape}"
    assert len(batch_head_targets) == 11, f"Expected 11 batch head targets, got {len(batch_head_targets)}"
    
    print("\n" + "=" * 60)
    print("TEST DATASET RESULT: PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()
