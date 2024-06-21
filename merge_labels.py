import json
import os

def merge_labels(original_file, new_file):
    # Load labels.json
    if os.path.exists(original_file):
        with open(original_file, 'r') as f:
            original_data = json.load(f)
    else:
        original_data = {"images": []}

    # Load new_labels.json
    if os.path.exists(new_file):
        with open(new_file, 'r') as f:
            new_data = json.load(f)
    else:
        new_data = {"images": []}

    # Create a dictionary to access images by file name for original data
    original_images_dict = {image["file"]: image for image in original_data["images"]}

    # Merge new labels into original labels
    for new_image in new_data["images"]:
        file_name = new_image["file"]
        if file_name in original_images_dict:
            original_annotations = original_images_dict[file_name]["annotations"]
            new_annotations = new_image["annotations"]
            # Merge annotations, avoiding duplicates
            for new_ann in new_annotations:
                if new_ann not in original_annotations:
                    original_annotations.append(new_ann)
        else:
            # If file name is not in original data, add the new image and its annotations
            original_data["images"].append(new_image)

    # Save the merged data back to the original file
    with open(original_file, 'w') as f:
        json.dump(original_data, f, indent=4)

if __name__ == "__main__":
    original_file = 'labels.json'
    new_file = 'new_labels.json'
    merge_labels(original_file, new_file)

    print(f"Labels from {new_file} merged into {original_file}")
