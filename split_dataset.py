import os
import shutil
import random

dataset_path = r"C:\Users\ABHISHEK BUDHWAT\OneDrive\Desktop\NEW_NUTRISNAP\Dataset\Indian_Food_Dataset"
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")

# Create train and valid folders if they don’t exist
for split in [train_path, valid_path]:
    os.makedirs(split, exist_ok=True)

# Split dataset (80% train, 20% validation)
split_ratio = 0.8  

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    
    if os.path.isdir(category_path):  # Ensure it's a folder
        images = os.listdir(category_path)
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        valid_images = images[split_index:]

        os.makedirs(os.path.join(train_path, category), exist_ok=True)
        os.makedirs(os.path.join(valid_path, category), exist_ok=True)

        # Move images
        for img in train_images:
            shutil.move(os.path.join(category_path, img), os.path.join(train_path, category, img))
        for img in valid_images:
            shutil.move(os.path.join(category_path, img), os.path.join(valid_path, category, img))

print("✅ Dataset successfully split into Train and Validation sets!")

