import os
import pickle
import numpy as np

ROOT_DIR = r"D:\AI DATASET"
UNIFIED_DIR = os.path.join(ROOT_DIR, "UPAR_UNIFIED")
ANNOTATIONS_DIR = os.path.join(UNIFIED_DIR, "annotations")
REPORTS_DIR = os.path.join(UNIFIED_DIR, "reports")

UNIFIED_PKL = os.path.join(ANNOTATIONS_DIR, "unified_annotations.pkl")
MISSING_TXT = os.path.join(REPORTS_DIR, "missing_images.txt")
VAL_REPORT_TXT = os.path.join(REPORTS_DIR, "validation_report.txt")

EXPECTED_ATTRS = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue",
    "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey",
    "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple",
    "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow",
    "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black",
    "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green",
    "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink",
    "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
    "LowerBody-Color-Yellow", "LowerBody-Color-Other", "LowerBody-Type-Trousers&Shorts",
    "LowerBody-Type-Skirt&Dress", "Accessory-Backpack", "Accessory-Bag",
    "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat"
]

def main():
    print("=" * 60)
    print("VALIDATING UNIFIED UPAR DATASET...")
    print("=" * 60)
    
    validation_passed = True
    issues = []
    
    if not os.path.exists(UNIFIED_PKL):
        print(f"ERROR: {UNIFIED_PKL} does not exist!")
        return
        
    with open(UNIFIED_PKL, "rb") as f:
        data = pickle.load(f)
        
    image_names = np.array(data["image_name"])
    labels = np.array(data["label"])
    dataset_ids = np.array(data["dataset_ids"])
    dataset_names = np.array(data["dataset_names"])
    partition = np.array(data["partition"])
    attr_names = list(data["attr_name"])
    
    total_images = len(image_names)
    print(f"Total images: {total_images}")
    
    # 1. Structure Check
    if total_images != 145656:
        issues.append(f"Image count mismatch: Expected 145656, got {total_images}")
        
    if labels.shape != (total_images, 40):
        issues.append(f"Labels shape mismatch: Expected ({total_images}, 40), got {labels.shape}")
        validation_passed = False
        
    if len(dataset_ids) != total_images or len(dataset_names) != total_images or len(partition) != total_images:
        issues.append("Length mismatch among annotations metadata arrays.")
        validation_passed = False

    # 2. Attribute Check
    if len(attr_names) != 40:
        issues.append(f"Attribute count mismatch: Expected 40, got {len(attr_names)}")
        validation_passed = False
    elif attr_names != EXPECTED_ATTRS:
        issues.append("Attribute order or names do not match expected 40 attributes.")
        validation_passed = False
    else:
        print("40 Attributes order check: PASS")

    # 3. RAP2 and Dataset IDs Check
    unique_ids = set(dataset_ids)
    if 3 in unique_ids:
        issues.append("RAP2 (dataset_id 3) is present in dataset!")
        validation_passed = False
    if unique_ids != {0, 1, 2}:
        issues.append(f"Unexpected dataset_ids present: {unique_ids}")
        validation_passed = False
    else:
        print("Dataset IDs check {0, 1, 2} (No RAP2): PASS")

    # 4. Label Values Check
    unique_labels = np.unique(labels)
    invalid_labels = [l for l in unique_labels if l not in [0, 1, 2]]
    if len(invalid_labels) > 0:
        issues.append(f"Invalid label values found: {invalid_labels}")
        validation_passed = False
    else:
        print(f"Label values check (unique: {unique_labels}): PASS")

    c0 = np.sum(labels == 0)
    c1 = np.sum(labels == 1)
    c2 = np.sum(labels == 2)
    print(f"  Count 0 (Negative): {c0}")
    print(f"  Count 1 (Positive): {c1}")
    print(f"  Count 2 (Unknown):  {c2}")

    # 5. Duplicates Check
    unique_paths_count = len(set(image_names))
    dup_paths = total_images - unique_paths_count
    
    basenames = [os.path.basename(p) for p in image_names]
    unique_basenames_count = len(set(basenames))
    dup_filenames = total_images - unique_basenames_count
    
    print(f"Duplicate paths: {dup_paths}")
    print(f"Duplicate filenames: {dup_filenames}")

    # 6. File System Disk Check
    print("\nIndexing disk files for verification...")
    disk_file_index = set()
    search_dirs = [
        r"D:\AI DATASET\3 Datasets",
        r"D:\AI DATASET",
        r"D:\AI_Project"
    ]
    for sdir in search_dirs:
        if os.path.exists(sdir):
            for root, dirs, files in os.walk(sdir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        disk_file_index.add(os.path.basename(f))
                        
    existing_count = 0
    missing_count = 0
    missing_paths = []
    missing_by_dataset = {"Market1501": 0, "PA100k": 0, "PETA": 0}

    for idx, rel_p in enumerate(image_names):
        fname = os.path.basename(rel_p)
        ds_name = dataset_names[idx]
        if fname in disk_file_index:
            existing_count += 1
        else:
            missing_count += 1
            missing_paths.append((ds_name, rel_p))
            missing_by_dataset[ds_name] = missing_by_dataset.get(ds_name, 0) + 1

    print(f"Disk check: Existing={existing_count}, Missing={missing_count}")

    # Write missing_images.txt if missing > 0
    with open(MISSING_TXT, "w", encoding="utf-8") as f:
        f.write(f"MISSING IMAGES REPORT\n")
        f.write(f"Total missing images: {missing_count}\n")
        f.write(f"Missing by dataset: {missing_by_dataset}\n\n")
        if missing_count > 0:
            f.write("Sample missing image paths:\n")
            for ds_n, p in missing_paths[:500]:
                f.write(f"[{ds_n}] {p}\n")
    print(f"Saved {MISSING_TXT}")

    # Write validation_report.txt
    status_str = "PASS" if validation_passed else "FAIL"
    with open(VAL_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("VALIDATION REPORT - UNIFIED UPAR DATASET\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Status: {status_str}\n")
        f.write(f"Total Samples: {total_images}\n")
        f.write(f"Labels Shape: {labels.shape}\n")
        f.write(f"Attribute Count: {len(attr_names)}\n")
        f.write(f"Dataset IDs Present: {sorted(list(unique_ids))}\n")
        f.write(f"RAP2 Included: NO\n")
        f.write(f"Label Values Present: {sorted(list(unique_labels))}\n")
        f.write(f"Existing Images on Disk: {existing_count}\n")
        f.write(f"Missing Images on Disk: {missing_count}\n")
        f.write(f"Duplicate Paths: {dup_paths}\n")
        f.write(f"Duplicate Filenames: {dup_filenames}\n\n")
        if issues:
            f.write("ISSUES FOUND:\n")
            for iss in issues:
                f.write(f"  - {iss}\n")
        else:
            f.write("No structural errors found.\n")
            
    print(f"Saved {VAL_REPORT_TXT}")
    print(f"\nVALIDATION STATUS: {status_str}")

if __name__ == "__main__":
    main()
