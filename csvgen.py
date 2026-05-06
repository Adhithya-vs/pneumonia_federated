"""
Run this script ONCE before training to generate the CSV label files.
Expected folder structure:

data/chest_xray/
  train_images/
    NORMAL/    *.jpg / *.jpeg / *.png
    PNEUMONIA/ *.jpg / *.jpeg / *.png
  val_images/
    NORMAL/
    PNEUMONIA/
  test_images/
    NORMAL/
    PNEUMONIA/
"""
import os
import csv

BASE_PATH = "data/chest_xray"

DATASETS = {
    "train": os.path.join(BASE_PATH, "train_images"),
    "val":   os.path.join(BASE_PATH, "val_images"),
    "test":  os.path.join(BASE_PATH, "test_images"),
}

# Multi-label: [pneumonia, covid, tuberculosis]
LABELS = {
    "NORMAL":    [0, 0, 0],
    "PNEUMONIA": [1, 0, 0],
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def create_csv(split_name, folder_path):
    csv_path = os.path.join(BASE_PATH, f"{split_name}_labels.csv")

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "pneumonia", "covid", "tuberculosis"])

        for class_name, label_row in LABELS.items():
            class_path = os.path.join(folder_path, class_name)

            if not os.path.exists(class_path):
                print(f"  ⚠️  Folder not found: {class_path}")
                continue

            count = 0
            for image_name in os.listdir(class_path):
                ext = os.path.splitext(image_name)[1].lower()
                if ext not in VALID_EXTENSIONS:
                    continue
                if not os.path.isfile(os.path.join(class_path, image_name)):
                    continue
                writer.writerow([f"{class_name}/{image_name}", *label_row])
                count += 1

            print(f"  {class_name}: {count} images")

    print(f"✅ Created: {csv_path}\n")


def main():
    for split_name, folder_path in DATASETS.items():
        print(f"Processing [{split_name}] ...")
        create_csv(split_name, folder_path)
    print("All CSV files created successfully.")


if __name__ == "__main__":
    main()