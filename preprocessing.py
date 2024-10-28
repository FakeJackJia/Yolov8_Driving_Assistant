import os
import json
import shutil

ann_folder = "train/ann"
image_folder = "train/img"
output_images_folder = "images"
output_labels_folder = "train"

class_map = {
    "car": 0,
    "pedestrian": 1,
    "cyclist": 2,
    "truck": 3
}

def convert_json_to_yolo(json_file, image_width, image_height):
    with open(json_file, 'r') as file:
        data = json.load(file)

    yolo_labels = []

    for obj in data["objects"]:
        class_title = obj["classTitle"]
        if class_title in class_map:
            x1, y1 = obj["points"]["exterior"][0]
            x2, y2 = obj["points"]["exterior"][1]

            x_center = (x1 + x2) / 2 / image_width
            y_center = (y1 + y2) / 2 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            class_id = class_map[class_title]
            yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return yolo_labels


for json_file in os.listdir(ann_folder):
    json_file_path = os.path.join(ann_folder, json_file)

    image_file_name = json_file.replace('.json', '')
    image_file_path = os.path.join(image_folder, image_file_name)

    with open(json_file_path, 'r') as file:
        data = json.load(file)
        image_width = data["size"]["width"]
        image_height = data["size"]["height"]

    yolo_labels = convert_json_to_yolo(json_file_path, image_width, image_height)

    yolo_file_name = json_file.replace('.png.json', '.txt')
    yolo_file_path = os.path.join(output_labels_folder, yolo_file_name)
    with open(yolo_file_path, 'w') as yolo_file:
        for label in yolo_labels:
            yolo_file.write(label + '\n')

    shutil.copy(image_file_path, output_images_folder)

    print(f"Processed: {json_file} -> {yolo_file_name} and copied image: {image_file_name}")