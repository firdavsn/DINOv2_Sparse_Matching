from utils import *
import numpy as np
from matplotlib import pyplot as plt
from transformers import SamModel, SamProcessor, SamConfig
import torch
import time
from patchify import patchify
from PIL import Image
from matplotlib.patches import Rectangle

patch_size = 256
original_dims = (512, 641)

def resize_image(img_arr: np.ndarray, w: int, h: int) -> np.ndarray:
    img = Image.fromarray(img_arr)
    img = img.resize((w, h))
    return np.array(img)

def load_model_and_processor(model_path) -> tuple[SamModel, SamProcessor]:
    # Load the model configuration
    model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    print(model_path)
    # Create an instance of the model architecture with the loaded configuration
    model = SamModel(config=model_config)
    # Update the model by loading the weights from saved file.
    model.load_state_dict(torch.load(model_path))
    
    return model, processor

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def prepare_image(image: np.ndarray, dims: tuple[int, int], verbose=True) -> torch.Tensor:
    start_time = time.time()
    
    image = resize_image(image, dims[0], dims[1])
    patches = patchify(np.array(image), (patch_size, patch_size, 3), step=256)
    patches = np.squeeze(patches)
    
    end_time = time.time()
    
    if verbose:
        print(f"Time taken to prepare image: {end_time - start_time:.2f} seconds")
    
    return patches

def predict(model: SamModel, processor: SamProcessor, device: torch.device, image: torch.Tensor, dims: tuple[int, int], thresh: float = 0.5, verbose=True) -> (torch.Tensor, torch.Tensor):
    start_time = time.time()
    model.eval()
    
    patches = prepare_image(image, dims, verbose=verbose)
    
    prob = None
    
    for row_patches in patches:
        row_prob = None
        for patch in row_patches:
            inputs = processor(patch, return_tensors="pt")
            inputs = {k:v.to(device) for k,v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False)
            single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
            single_patch_prob = single_patch_prob / np.sum(single_patch_prob)
            
            if row_prob is None:
                row_prob = single_patch_prob
            else:
                row_prob = np.concatenate((row_prob, single_patch_prob), axis=1)
                
        if prob is None:
            prob = row_prob
        else:
            prob = np.concatenate((prob, row_prob), axis=0)
    
    # # Find top 4 patches with highest stddev
    # highest_stddev = []
    # for i in range(int(prob.shape[0] / patch_size)):
    #     for j in range(int(prob.shape[1] / patch_size)):
    #         patch = prob[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
    #         stddev = np.std(patch)
    #         highest_stddev.append((i, j, stddev))
    # highest_stddev = sorted(highest_stddev, key=lambda x: x[2], reverse=True)[:4]
    # print([(i, j) for i, j, _ in highest_stddev])
    
    # # Zero patches not in highest_stddev
    # for i in range(int(prob.shape[0] / patch_size)):
    #     for j in range(int(prob.shape[1] / patch_size)):
    #         if (i, j) not in [(i, j) for i, j, _ in highest_stddev]:
    #             prob[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 0
    
    pred = (prob > prob.max() * thresh).astype(np.uint8)
    
    pred = resize_image(pred, original_dims[1], original_dims[0])
    prob = resize_image(prob, original_dims[1], original_dims[0])
    
    end_time = time.time()
    if verbose:
        print(f"Time taken to predict: {end_time - start_time:.2f} seconds")
    
    return pred, prob

def plot_prediction(image: np.ndarray, prediction: np.ndarray, probability: np.ndarray, idx: int, bbox: tuple = None) -> None:
    if bbox:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fix, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the first image on the left
    axes[0].imshow(np.array(image), cmap='gray')  # Assuming the first image is grayscale
    axes[0].set_title(idx)

    # Plot the second image on the right
    axes[1].imshow(probability)  # Assuming the second image is grayscale
    axes[1].set_title("Probability Map")

    # Plot the second image on the right
    axes[2].imshow(prediction, cmap='gray')  # Assuming the second image is grayscale
    axes[2].set_title("Prediction")
    
    if bbox:
        axes[3].imshow(prediction, cmap='gray') 
        axes[3].set_title("Prediction with bbox")
        
        (x, y), w, h = bbox
        r = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axes[3].add_patch(r)

    # Hide axis ticks and labels
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Display the images side by side
    plt.show()


images = get_images('test_runtime/images/')
masks = get_images('test_runtime/masks/')
bboxes = get_bboxes('test_runtime/bboxes/')

device = get_device()

models = {}
processors = {}

model_paths = {
    512: "sam_checkpoints_finetuned/512.pth",
    1024: "sam_checkpoints_finetuned/1024.pth",
    2048: "sam_checkpoints_finetuned/2048.pth"

}

for size in [512, 1024, 2048]:
    model, processor = load_model_and_processor(model_paths[size])
    model.to(device)
    models[size] = model
    processors[size] = processor
    
torch.cuda.empty_cache()

times = {
    512: [],
    1024: [],
    2048: []
}

for key in tqdm(list(images.keys())[:10]):
    for size in [512, 1024, 2048]:
        start_time = time.time()
        predict(models[size], processors[size], device, images[key], (size, size), thresh=0.5, verbose=False)
        end_time = time.time()
        times[size].append(end_time - start_time)

for size in [512, 1024, 2048]:
    print(f"Average time taken for {size}x{size} image: {np.mean(times[size]):.2f} seconds")