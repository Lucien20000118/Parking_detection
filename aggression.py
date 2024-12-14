import kagglehub
import os
import shutil
import json
import re
import math

def download_dataset(dataset_path):
    print("Downloading dataset...")
    path = kagglehub.dataset_download("ammarnassanalhajali/pklot-dataset")
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)  # Remove existing folder
    shutil.copytree(path, dataset_path, dirs_exist_ok=True)
    print("Dataset downloaded to:", dataset_path)

def process_annotations(dataset_path, output_labels_dirname, target_images_dirname):
    print("Processing annotations and images...")
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith("_annotations.coco.json"):
                json_file = os.path.join(root, file)
                output_labels_dir = os.path.join(root, output_labels_dirname)
                target_images_dir = os.path.join(root, target_images_dirname)
                os.makedirs(output_labels_dir, exist_ok=True)
                os.makedirs(target_images_dir, exist_ok=True)

                with open(json_file, "r") as f:
                    data = json.load(f)

                images = {image["id"]: image["file_name"] for image in data["images"]}
                annotations = data["annotations"]

                for image_id, file_name in images.items():
                    process_single_image(image_id, file_name, annotations, data, output_labels_dir, root, target_images_dir)

def process_single_image(image_id, file_name, annotations, data, output_labels_dir, root, target_images_dir):
    label_file = os.path.join(output_labels_dir, f"{file_name.split('.')[0]}.txt")
    image_annotations = [ann for ann in annotations if ann["image_id"] == image_id]
    with open(label_file, "w") as lf:
        for ann in image_annotations:
            category_id = ann["category_id"] - 1
            bbox = ann["bbox"]
            angle = calculate_angle(bbox)

            # Convert bbox to YOLO format
            x_center = (bbox[0] + bbox[2] / 2) / data["images"][0]["width"]
            y_center = (bbox[1] + bbox[3] / 2) / data["images"][0]["height"]
            width = bbox[2] / data["images"][0]["width"]
            height = bbox[3] / data["images"][0]["height"]

            lf.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {angle:.6f}\n")

    source_path = os.path.join(root, file_name)
    target_path = os.path.join(target_images_dir, file_name)
    if os.path.exists(source_path):
        shutil.move(source_path, target_path)

def calculate_angle(bbox):
    x_min, y_min, width, height = bbox
    x1, y1 = x_min, y_min
    x2, y2 = x_min + width, y_min
    dx, dy = x2 - x1, y2 - y1
    angle = math.atan2(dy, dx)
    return abs(angle) % (math.pi / 2)

def rename_images(dataset_path):
    print("Renaming image files...")
    for root, dirs, files in os.walk(dataset_path):
        if "images" in dirs:
            images_folder = os.path.join(root, "images")
            for file_name in os.listdir(images_folder):
                old_file_path = os.path.join(images_folder, file_name)
                if os.path.isfile(old_file_path) and file_name.endswith(".jpg"):
                    new_file_name = re.sub(r"\.rf\.[a-f0-9]+", "", file_name)
                    new_file_path = os.path.join(images_folder, new_file_name)
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: {old_file_path} -> {new_file_path}")

def main():
    dataset_path = "./pklot_dataset"
    output_labels_dirname = "labels"
    target_images_dirname = "images"

    download_dataset(dataset_path)
    process_annotations(dataset_path, output_labels_dirname, target_images_dirname)
    rename_images(dataset_path)
    print("Processing complete!")

if __name__ == "__main__":
    main()
