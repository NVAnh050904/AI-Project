import os
import pickle
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = r"D:\AI DATASET"
UNIFIED_PKL = os.path.join(ROOT_DIR, "UPAR_UNIFIED", "annotations", "unified_annotations.pkl")

def main():
    if not os.path.exists(UNIFIED_PKL):
        print(f"File not found: {UNIFIED_PKL}")
        return

    with open(UNIFIED_PKL, "rb") as f:
        data = pickle.load(f)

    image_names = np.array(data["image_name"])
    labels = np.array(data["label"])
    dataset_ids = np.array(data["dataset_ids"])
    dataset_names = np.array(data["dataset_names"])
    partition = np.array(data["partition"])
    attr_names = list(data["attr_name"])

    m_cnt = np.sum(dataset_ids == 0)
    p_cnt = np.sum(dataset_ids == 1)
    pe_cnt = np.sum(dataset_ids == 2)
    total_cnt = len(image_names)

    c0 = np.sum(labels == 0)
    c1 = np.sum(labels == 1)
    c2 = np.sum(labels == 2)

    tr_cnt = np.sum(partition == "train")
    va_cnt = np.sum(partition == "val")
    te_cnt = np.sum(partition == "test")

    rap2_in = 3 in set(dataset_ids)
    invalid_l = [l for l in np.unique(labels) if l not in [0, 1, 2]]

    # Disk check count
    disk_file_index = set()
    for sdir in [r"D:\AI DATASET\3 Datasets", r"D:\AI DATASET", r"D:\AI_Project"]:
        if os.path.exists(sdir):
            for root, dirs, files in os.walk(sdir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        disk_file_index.add(os.path.basename(f))

    existing_cnt = sum(1 for name in image_names if os.path.basename(name) in disk_file_index)
    missing_cnt = total_cnt - existing_cnt
    dup_paths = total_cnt - len(set(image_names))

    print("\n" + "="*40)
    print("UNIFIED UPAR DATASET")
    print("="*40)
    print(f"Market1501 : {m_cnt:,}")
    print(f"PA100k     : {p_cnt:,}")
    print(f"PETA       : {pe_cnt:,}")
    print("-" * 40)
    print(f"TOTAL      : {total_cnt:,}")
    print(f"\nAttributes : {len(attr_names)}")
    print("\nLabel values:")
    print(f"0 = {c0:,}")
    print(f"1 = {c1:,}")
    print(f"2 = {c2:,}")
    print(f"\nTrain : {tr_cnt:,}")
    print(f"Val   : {va_cnt:,}")
    print(f"Test  : {te_cnt:,}")
    print(f"\nMissing images : {missing_cnt}")
    print(f"Duplicate      : {dup_paths}")
    print(f"Invalid labels : {len(invalid_l)}")
    print(f"\nRAP2 included  : {'YES' if rap2_in else 'NO'}")
    print("="*40)
    print(f"VALIDATION: {'PASS' if (not rap2_in and len(invalid_l) == 0 and total_cnt == 145656) else 'FAIL'}")
    print("="*40)

    print("\n" + "="*85)
    print("ATTRIBUTE BREAKDOWN IN UNIFIED DATASET")
    print("="*85)
    print(f"| {'ID':<2} | {'Attribute Name':<32} | {'Label 1':<10} | {'Label 0':<10} | {'Label 2':<10} |")
    print("|----|----------------------------------|------------|------------|------------|")
    for idx, name in enumerate(attr_names):
        col = labels[:, idx]
        l1 = np.sum(col == 1)
        l0 = np.sum(col == 0)
        l2 = np.sum(col == 2)
        print(f"| {idx:02d} | {name:<32} | {l1:<10} | {l0:<10} | {l2:<10} |")

if __name__ == "__main__":
    main()
