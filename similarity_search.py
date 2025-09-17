import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use('Qt5Agg')
import faiss
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import matplotlib.pyplot as plt
if __name__ == "__main__":
    image = Image.open('uploads101')

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
    print('Model loaded')
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0]
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    faiss.normalize_L2(vector)
    print('Features extracted')
    index = faiss.read_index("index/vector.index")
    print('Index loaded')
    print("About to search index")
    try:
        d, i = index.search(vector, 15)
        print("Search completed")
    except Exception as e:
        print("Error during search:", e)
    print('distances:', d)
    print('indexes:', i)
    catalogue_images = []
    for root, dirs, files in os.walk('./catalogue101'):
        for file in files:
            if file.endswith('jpg'):
                catalogue_images.append(root + '/' + file)

    result_paths = [catalogue_images[idx] for idx in i[0]]
    scores = d[0]

    fig, axes = plt.subplots(4, 5, figsize=(18, 12))
    axes = axes.flatten()

    axes[0].imshow(image)
    axes[0].set_title('Query Image')
    axes[0].axis('off')

    for j in range(15):
        img = Image.open(result_paths[j])
        axes[j+1].imshow(img)
        axes[j+1].set_title(f'Score: {scores[j]:.3f}')
        axes[j+1].axis('off')

    for k in range(16, 20):
        axes[k].axis('off')

    plt.tight_layout()
    plt.savefig('results101/result18.png')