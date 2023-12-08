# src/dinov2_custom/sparse_matcher.py

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

REPO_NAME = "facebookresearch/dinov2"
MODEL_NAMES = {"large" : "dinov2_vitg14",
               "medium": "dinov2_vitl14",
               "small" : "dinov2_vitb14",
               "tiny"  : "dinov2_vits14"}

class DINOv2:
    """
    DINOv2 class uses Meta's DINOv2 model to extract feature embeddings of an image.
    
    Attributes:
        smaller_edge_size (int): Size of the smaller edge of the image for processing.
        half_precision (bool): Flag to indicate if half precision computation is used.
        device (str): Computing device, e.g., 'cuda' or 'cpu'.
        model (torch.Module): Loaded DINO model.
        transform (torchvision.transforms): Transformations applied to the input image.
    """
    
    def __init__(self, 
                 model_size: str = "small",
                 smaller_edge_size: int = 448, 
                 half_precision: bool = False, 
                 device: str = "cuda"):
        """
        Initializes the Sparse_Matcher with specified parameters and loads the DINO model.
        
        Arguments:
            model_size (str, "small"): Size of the DINO model to load in terms of number of params
                'large' = 1.1 B 
                'medium' = 300 M
                'small' = 86 M
                'tiny' = 21 M
            smaller_edge_size (int, 448): Size of the smaller edge of the image for processing.
            half_precision (bool, False): Whether to use half precision computation.
            device (str, "cuda"): The computing device, e.g., 'cuda' or 'cpu'.
        """
        
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.device = device
        
        # Loading the DINO model with optional half precision.
        self.model = torch.hub.load(repo_or_dir=REPO_NAME, model=MODEL_NAMES[model_size])
        if self.half_precision:
            self.model = self.model.half()  # Convert to half precision if enabled
        self.model = self.model.to(self.device)  # Move model to specified device
        self.model.eval()  # Set model to evaluation mode

        # Transformations for input image preprocessing.
        self.transform = transforms.Compose([
            transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet defaults
            ])

    def prepare_image(self, rgb_image_numpy: np.ndarray) -> (torch.Tensor, tuple[int], float):
        """
        Prepares an RGB image for processing by resizing and cropping to fit the model's requirements.
        
        Arguments:
            rgb_image_numpy (numpy.ndarray): The RGB image in NumPy array format.

        Returns:
            Tuple containing the processed image tensor, grid size, and resize scale.
        """
    
        # Convert NumPy array to PIL image and apply transformations.
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]  # Calculate scale of resize

        # Cropping the image to fit the model's input requirements.
        height, width = image_tensor.shape[1:]  # Extracting height and width
        # Ensure dimensions are multiples of the patch size.
        cropped_width = width - width % self.model.patch_size
        cropped_height = height - height % self.model.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        # Calculate grid size based on the cropped image dimensions.
        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size, resize_scale

    def prepare_mask(self, mask_image_numpy: np.ndarray, grid_size: tuple[int], resize_scale: float) -> np.ndarray:
        """
        Prepares a mask image for processing, aligning it with the dimensions of the processed main image.

        Arguments:
            mask_image_numpy (numpy.ndarray): The mask image in NumPy array format.
            grid_size (tuple[int]): The grid size of the processed main image.
            resize_scale (float): The scale at which the main image was resized.

        Returns:
            NumPy array of the resized mask.
        """
    
        # Crop and resize mask to align with the processed image's grid.
        cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]
        image = Image.fromarray(cropped_mask_image_numpy)
        resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
        resized_mask = np.asarray(resized_mask).flatten()  # Flatten the resized mask
        return resized_mask

    def extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Extracts features from an image tensor using the DINOv2 model.

        Arguments:
            image_tensor (torch.Tensor): The image tensor to extract features from.

        Returns:
            NumPy array of extracted features.
        """

    
        # Perform inference without gradient calculation for efficiency.
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            # Extracting features (tokens) from the image.
            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        return tokens.cpu().numpy()  # Return the extracted features as a NumPy array


    def idx_to_source_position(self, idx: int, grid_size: tuple, resize_scale: float) -> (int, int):
        """
        Converts an index in the flattened feature map back to its original position in the source image.

        Arguments:
            idx (int): The index in the flattened feature map.
            grid_size (tuple): The grid size of the processed image.
            resize_scale (float): The scale at which the original image was resized.

        Returns:
            Tuple of row and column indicating the position in the original image.
        """

        # Calculating the row and column in the original image from the index.
        row = (idx // grid_size[1]) * self.model.patch_size * resize_scale + self.model.patch_size / 2
        col = (idx % grid_size[1]) * self.model.patch_size * resize_scale + self.model.patch_size / 2
        return row, col
        
    def get_embedding_visualization(self, tokens: np.ndarray, grid_size: tuple, resized_mask: np.ndarray = None) -> np.ndarray:
        """
        Generates a visualization of the embeddings using PCA.

        Arguments:
            tokens (numpy.ndarray): The feature tokens extracted from the image.
            grid_size (tuple): The grid size of the processed image.
            resized_mask (numpy.ndarray, optional): The resized mask for selecting specific tokens.

        Returns:
            NumPy array representing the PCA-reduced and normalized tokens for visualization.
        """
    
        # Applying PCA to reduce the feature dimensions for visualization.
        pca = PCA(n_components=3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]  # Apply mask if provided
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        
        # Reformatting tokens for visualization based on the resized mask.
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        
        # Normalizing tokens for better visualization.
        normalized_tokens = (reduced_tokens - np.min(reduced_tokens)) / (np.max(reduced_tokens) - np.min(reduced_tokens))
        return normalized_tokens
