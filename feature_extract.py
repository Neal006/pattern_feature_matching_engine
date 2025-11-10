
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import faiss
import numpy as np
import os
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast=True)
model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

images = []
for root, dirs, files in os.walk('./catalogue'):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(root  + '/'+ file)

def add_vector_to_index(embedding, index):
    vector = embedding.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    index.add(vector)
index = faiss.IndexFlatL2(384)

image_paths = []
start_time = time.time()
total_images = len(images)

for image_path in tqdm(images):
    img = Image.open(image_path).convert('RGB')
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0]  
    add_vector_to_index(features, index)
    image_paths.append(image_path)

end_time = time.time()
total_time = end_time - start_time
avg_time_per_image = total_time / total_images

print(f"\nFeature Extraction Statistics:")
print(f"Total time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Number of images processed: {total_images}")
print(f"Average time per image: {avg_time_per_image:.3f} seconds")
print(f"Using device: {device}")

faiss.write_index(index, "index/vector.index")

with open("index/vector.index.paths.txt", "w") as f:
    for path in image_paths:
        f.write(path + "\n")

input("Press Enter to exit...")