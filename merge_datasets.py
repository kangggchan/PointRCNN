#!/usr/bin/env python3
"""
Merge two KITTI datasets (scala2 and custom) into a single consolidated dataset.
- Renames all merged samples to unified XXXXXX.bin / XXXXXX.txt format
- Keeps only paired .bin/.txt samples
- Creates a new combined dataset in data/scala2_combined/
"""

import shutil
from pathlib import Path

def get_frame_number(filename):
    """Extract frame number from filename."""
    base = filename.rsplit('.', 1)[0]  # Remove extension
    if base.startswith('frame_'):
        return int(base[6:])
    else:
        try:
            return int(base)
        except:
            return None


def build_paired_entries(bin_dir, labels_dir):
    bin_files = sorted(bin_dir.glob("*.bin"))
    label_files = sorted(labels_dir.glob("*.txt"))

    bin_map = {f.stem: f for f in bin_files}
    label_map = {f.stem: f for f in label_files}

    paired_stems = sorted(set(bin_map.keys()) & set(label_map.keys()), key=lambda s: get_frame_number(f"{s}.bin") if get_frame_number(f"{s}.bin") is not None else s)
    paired_entries = [(stem, bin_map[stem], label_map[stem]) for stem in paired_stems]

    unpaired_bin_count = len(bin_files) - len(paired_entries)
    unpaired_label_count = len(label_files) - len(paired_entries)

    return paired_entries, unpaired_bin_count, unpaired_label_count

def main():
    script_dir = Path(__file__).parent
    
    source_dataset = script_dir / "data" / "scala2"
    target_dataset = script_dir / "data" / "custom"
    merged_dataset = script_dir / "data" / "dataset"
    
    # Validate source directories
    source_bin = source_dataset / "bin"
    source_labels = source_dataset / "labels"
    target_bin = target_dataset / "bin"
    target_labels = target_dataset / "labels"
    
    for d in [source_bin, source_labels, target_bin, target_labels]:
        if not d.exists():
            print(f"ERROR: Directory not found: {d}")
            return False
    
    # Create merged dataset directories
    merged_bin = merged_dataset / "bin"
    merged_labels = merged_dataset / "labels"
    merged_bin.mkdir(parents=True, exist_ok=True)
    merged_labels.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DATASET MERGE TOOL")
    print("=" * 70)
    print(f"Source (scala2):    {source_bin}")
    print(f"Target (custom):    {target_bin}")
    print(f"Merged output:      {merged_bin}")
    print()
    
    # **Step 1**: Copy source (scala2) files (only paired .bin/.txt), rename to XXXXXX
    print("[1/4] Copying source (scala2) dataset...")
    source_pairs, source_unpaired_bin, source_unpaired_label = build_paired_entries(source_bin, source_labels)

    next_merged_id = 0
    for i, (_, bin_file, label_file) in enumerate(source_pairs):
        if i % 100 == 0:
            print(f"  Processing pair {i+1}/{len(source_pairs)}...")

        new_stem = f"{next_merged_id:06d}"
        shutil.copy2(bin_file, merged_bin / f"{new_stem}.bin")
        shutil.copy2(label_file, merged_labels / f"{new_stem}.txt")
        next_merged_id += 1

    print(f"  Copied {len(source_pairs)} paired samples")
    if source_unpaired_bin > 0 or source_unpaired_label > 0:
        print(f"  ⚠ Skipped {source_unpaired_bin} unpaired .bin, {source_unpaired_label} unpaired .txt files")
    print()
    
    # **Step 2**: Rename and copy target (custom) files to XXXXXX (only paired .bin/.txt)
    print("[2/4] Renaming and copying target (custom) dataset...")
    target_pairs, target_unpaired_bin, target_unpaired_label = build_paired_entries(target_bin, target_labels)

    for i, (_, bin_file, label_file) in enumerate(target_pairs):
        if i % 500 == 0:
            print(f"  Processing pair {i+1}/{len(target_pairs)}...")

        new_stem = f"{next_merged_id:06d}"
        shutil.copy2(bin_file, merged_bin / f"{new_stem}.bin")
        shutil.copy2(label_file, merged_labels / f"{new_stem}.txt")
        next_merged_id += 1

    print(f"  Renamed and copied {len(target_pairs)} paired samples")
    if target_unpaired_bin > 0 or target_unpaired_label > 0:
        print(f"  ⚠ Skipped {target_unpaired_bin} unpaired .bin, {target_unpaired_label} unpaired .txt files")
    print()
    
    # **Step 3**: Verify merged dataset
    print("[3/4] Verifying merged dataset...")
    final_bin_files = list(merged_bin.glob("*.bin"))
    final_label_files = list(merged_labels.glob("*.txt"))
    
    print(f"  Total .bin files in merged dataset: {len(final_bin_files)}")
    print(f"  Total .txt files in merged dataset: {len(final_label_files)}")
    print()
    
    # **Step 4**: Summary and recommendations
    print("[4/4] Summary")
    print("=" * 70)
    expected_total = len(source_pairs) + len(target_pairs)
    print(f"Expected total .bin files:  {expected_total}")
    print(f"Actual total .bin files:    {len(final_bin_files)}")
    if len(final_bin_files) == expected_total:
        print("✓ SUCCESS: All paired files merged correctly!")
    else:
        print("✗ WARNING: File count mismatch!")
    print()
    print("To use the merged dataset, update your scripts to use:")
    print(f"  --data_root {merged_dataset.relative_to(script_dir)}")
    print()
    print("Original datasets:")
    print(f"  {source_dataset.relative_to(script_dir)} (scala2)")
    print(f"  {target_dataset.relative_to(script_dir)} (custom)")
    print()
    print("You can safely delete the original datasets after verification.")
    print("=" * 70)
    
    return len(final_bin_files) == expected_total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
