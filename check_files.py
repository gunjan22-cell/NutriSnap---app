import os

dataset_path = r"C:\Users\ABHISHEK BUDHWAT\OneDrive\Desktop\NEW_NUTRISNAP\Dataset\Indian_Food_Dataset"
# Count total files
total_files = sum([len(files) for _, _, files in os.walk(dataset_path)])

if total_files == 0:
    print("⚠️ WARNING: No images found. Check the dataset location.")
else:
    print(f"✅ Total files found: {total_files}")

