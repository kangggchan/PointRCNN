#!/usr/bin/env python
import os
import glob
from collections import Counter

label_dir = '../data/dataset/KITTI/object/training/label_2'
class_counts = Counter()

for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cls = parts[0]
                class_counts[cls] += 1

print('Class distribution in training labels:')
for cls, count in sorted(class_counts.items()):
    print(f'{cls}: {count}')