import os
import shutil
import pandas as pd

dataset_dir = 'dataset'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
train_csv = os.path.join(dataset_dir, 'training_set.csv')

df = pd.read_csv(train_csv)

for _, row in df.iterrows():
    filename = row['filename']
    label = row['label']
    src_path = os.path.join(train_dir, filename)
    label_dir = os.path.join(train_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    dst_path = os.path.join(label_dir, filename)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)

print("Images organized into label-wise folders.")
