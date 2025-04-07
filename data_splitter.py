import os
import shutil
import random


dataset_dir = 'dataset/inaturalist_12K'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')

# 'val' → 'test' [renaming as per assignment link]
if os.path.exists(val_dir):
    os.rename(val_dir, test_dir)
    print(f"Renamed 'val' → 'test'")
else:
    print(f"'val' folder not found, skipping rename.")


new_val_dir = os.path.join(dataset_dir, 'val')
os.makedirs(new_val_dir, exist_ok=True)
random.seed(42)

# Walk through each class folder in train
for class_name in os.listdir(train_dir):
    class_train_path = os.path.join(train_dir, class_name)
    class_val_path = os.path.join(new_val_dir, class_name)

    if not os.path.isdir(class_train_path):
        continue

    os.makedirs(class_val_path, exist_ok=True)
    images = os.listdir(class_train_path)
    random.shuffle(images)

    # Split: 20% for val
    val_count = int(0.2 * len(images))
    val_images = images[:val_count]

    for img_name in val_images:
        src_path = os.path.join(class_train_path, img_name)
        dst_path = os.path.join(class_val_path, img_name)
        shutil.move(src_path, dst_path)
    print(f"Moved {val_count} images from {class_name} to val.")


print("Dataset partitioning complete.")
