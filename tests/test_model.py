import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.upar.loader import UPARDataset, get_upar_transforms
from models.hydraplus.par_model import UnifiedPARModel

def main():
    print("=" * 60)
    print("TEST: TESTING MULTI-HEAD MODEL FORWARD PASS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_transform, _ = get_upar_transforms(height=256, width=128)
    dataset = UPARDataset(split='train', transform=train_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    images, raw_labels, head_targets, _, _ = next(iter(loader))
    images = images.to(device)
    raw_labels = raw_labels.to(device)
    
    print(f"Input images shape: {images.shape}")
    print(f"Input labels shape: {raw_labels.shape}")
    
    model = UnifiedPARModel(num_attributes=40, backbone_name='resnet50', pretrained=True).to(device)
    model.eval()
    
    with torch.no_grad():
        outputs_dict = model(images)
        probs_40 = model.predict_40_probabilities(outputs_dict)
        logits_40 = model.predict_40_logits(outputs_dict)
        
    print(f"Model outputs dict keys ({len(outputs_dict)} heads): {list(outputs_dict.keys())}")
    for h_name, h_logits in outputs_dict.items():
        print(f"  - Head '{h_name}': shape {h_logits.shape}")
        
    print(f"Predicted 40 probabilities shape: {probs_40.shape}")
    print(f"Predicted 40 logits shape: {logits_40.shape}")
    
    assert len(outputs_dict) == 11, f"Expected 11 heads, got {len(outputs_dict)}"
    assert outputs_dict["age"].shape == torch.Size([8, 3]), f"Expected age head shape [8, 3], got {outputs_dict['age'].shape}"
    assert outputs_dict["upper_color"].shape == torch.Size([8, 12]), f"Expected upper_color head shape [8, 12], got {outputs_dict['upper_color'].shape}"
    assert probs_40.shape == torch.Size([8, 40]), f"Expected probs_40 shape [8, 40], got {probs_40.shape}"
    assert logits_40.shape == torch.Size([8, 40]), f"Expected logits_40 shape [8, 40], got {logits_40.shape}"
    
    print("\n" + "=" * 60)
    print("TEST MODEL RESULT: PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()
