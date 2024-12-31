import os
import pickle
import json
from collections import Counter
from PIL import Image
from sklearn.model_selection import train_test_split


def build_vocab(captions_file):
    vocab = Counter()
    with open(captions_file, 'r', encoding='utf-8') as file:
        next(file)  # Skip the header row
        for line in file:
            line = line.strip()
            if ',' not in line:  # Skip invalid lines
                print(f"Skipping invalid line: {line}")
                continue
            _, caption = line.split(',', 1)  # Use comma as separator
            tokens = caption.lower().split()
            vocab.update(tokens)
    return vocab


def prepare_dataset(captions_file, images_folder, output_folder, test_size=0.2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Build vocabulary
    vocab = build_vocab(captions_file)

    # Save vocabulary
    with open(os.path.join(output_folder, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    print("Vocabulary saved to vocab.pkl")

    # Split dataset into train/test
    image_caption_pairs = []
    with open(captions_file, 'r', encoding='utf-8') as file:
        next(file)  # Skip header
        for line in file:
            line = line.strip()
            if ',' not in line:
                continue
            image, caption = line.split(',', 1)
            image_caption_pairs.append((image, caption))

    train_pairs, test_pairs = train_test_split(image_caption_pairs, test_size=test_size, random_state=42)

    # Save train/test splits
    with open(os.path.join(output_folder, 'train_split.pkl'), 'wb') as f:
        pickle.dump(train_pairs, f)
    with open(os.path.join(output_folder, 'test_split.pkl'), 'wb') as f:
        pickle.dump(test_pairs, f)
    print("Train and test splits saved.")

    # Create prepared_data.json
    prepared_data = []
    for image_name, caption in image_caption_pairs:
        image_path = os.path.join(images_folder, image_name)
        if os.path.exists(image_path):
            prepared_data.append({"image": image_path, "caption": caption})
        else:
            print(f"Image not found: {image_path}")

    with open(os.path.join(output_folder, 'prepared_data.json'), 'w', encoding='utf-8') as f:
        json.dump(prepared_data, f, indent=4)
    print("Prepared data saved to prepared_data.json.")

    # Resize and save images in the output folder
    for image_name, _ in image_caption_pairs:
        image_path = os.path.join(images_folder, image_name)
        if not os.path.exists(image_path):
            continue

        try:
            with Image.open(image_path) as img:
                img = img.resize((224, 224))  # Resize to 224x224
                img.save(os.path.join(output_folder, image_name))
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")


if __name__ == "__main__":
    # captions_file = "C:/Users/Ansh Varshney/Desktop/Image_Captioning/dataset/captions.txt"
    # images_folder = "C:/Users/Ansh Varshney/Desktop/Image_Captioning/dataset/Images"
    # output_folder = "C:/Users/Ansh Varshney/Desktop/Image_Captioning/dataset/processed"

    # prepare_dataset(captions_file, images_folder, output_folder)

# Paths
    captions_file = "C:/Users/Ansh Varshney/Desktop/Image_Captioning/dataset/captions.txt"
    output_file = "C:/Users/Ansh Varshney/Desktop/Image_Captioning/dataset/processed/prepared_data.json"

# Initialize a list to store the image-caption mappings
    data = []

# Read the captions.txt file
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
        # Split each line into image filename and caption
            parts = line.strip().split(',', 1)  # Split into two parts: filename, caption
            if len(parts) == 2:  # Ensure there are two parts
                image_file = parts[0]
                caption = parts[1]
                data.append([image_file, caption])

# Write the data to prepared_data.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"JSON file successfully created at {output_file}")

