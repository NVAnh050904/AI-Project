import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.upar.loader import UPARDataset, get_upar_transforms, build_batch_multi_head_targets
from models.hydraplus.par_model import UnifiedPARModel
from training.loss import MultiHeadPARLoss

def main():
    print("=" * 60)
    print("TEST: TESTING MULTI-HEAD FORWARD + LOSS + BACKWARD")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_transform, _ = get_upar_transforms(height=256, width=128)
    dataset = UPARDataset(split='train', transform=train_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    batch = next(iter(loader))
    images = batch[0].to(device)
    raw_labels = batch[1].to(device)
    head_targets = build_batch_multi_head_targets(raw_labels)
    
    # 1. Forward Pass
    model = UnifiedPARModel(num_attributes=40, backbone_name='resnet50', pretrained=True).to(device)
    model.train()
    
    outputs_dict = model(images)
    print(f"Forward Pass: PASS (outputs dict keys: {list(outputs_dict.keys())})")
    assert len(outputs_dict) == 11, "Expected 11 head outputs!"
    
    # 2. Loss Computation
    criterion = MultiHeadPARLoss().to(device)
    total_loss, head_losses = criterion.compute_losses(outputs_dict, head_targets)
    print(f"Loss Pass: PASS (total_loss: {total_loss.item():.4f})")
    for h_name, h_val in head_losses.items():
        print(f"  - {h_name:<15} loss: {h_val.item():.4f}")
        assert not torch.isnan(h_val) and not torch.isinf(h_val), f"Loss for head '{h_name}' is NaN or Inf!"
        
    assert not torch.isnan(total_loss) and not torch.isinf(total_loss), "Total loss is NaN or Inf!"
    
    # 3. Backward Pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    total_loss.backward()
    
    grad_count = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_count += 1
    print(f"Backward Pass: PASS ({grad_count} parameters received gradients)")
    assert grad_count > 0, "No gradients computed!"
    
    optimizer.step()
    
    print("\n" + "=" * 60)
    print("TEST TRAINING RESULT: ALL 3 PASS (Forward: PASS, Loss: PASS, Backward: PASS)")
    print("=" * 60)

if __name__ == "__main__":
    main()
