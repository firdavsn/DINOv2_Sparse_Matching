# src/dinov2_custom/segmenter.py

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

CHECKPOINTS = {"large" : r"...\\sam_checkpoints\\sam_vit_h_4b8939.pth",
               "medium": r"...\\sam_checkpoints\\sam_vit_l_0b3195.pth",
               "small" : r"...\\sam_checkpoints\\sam_vit_b_01ec64.pth"}

MODEL_TYPES = {"large" : "vit_h",
               "medium": "vit_l",
               "small" : "vit_b"}

class Segmenter:
    """
    Segmenter class to apply semantic segmentation to images using Meta's Segment Anything Model (SAM).

    Attributes:
        model (torch.Module): The loaded SAM model for image segmentation.
        mask_generator (SamAutomaticMaskGenerator): Utility for generating masks from the SAM model.
        device (str): The computing device used for model operations, e.g., 'cuda' or 'cpu'.
    """
    
    def __init__(self, sam_checkpoint_path: str = None, model_type: str = None, model_size: str = "large", device: str = "cuda"):
        """
        Initialize Segmenter model and any utils.

        Arguments:
            sam_checkpoint_path (str, optional): Path to SAM checkpoint for model.
            model_type (str, optional): Type of model in SAM checkpoint.
            model_size (str, "large"): Size of the model, options are 'large', 'medium', or 'small'.
            device (str, "cuda"): Device to run on; options are 'cuda' or 'cpu'.
        """
        # Determine pretrained model checkpoint and model type
        checkpoint, model_type = None, model_type
        if sam_checkpoint_path and model_type:
            checkpoint = sam_checkpoint_path
        else:
            checkpoint = CHECKPOINTS[model_size]
            model_type = MODEL_TYPES[model_size]
        
        # Load model
        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.model.to(device=device)
        
        # Initialize mask generator
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        
    def generate_masks(self, image: np.ndarray) -> list[dict]:
        """
        Generates masks of the image.

        Arguments:
            image (np.ndarray): Image represented in an array.

        Returns:
            list[dict]: A list of dictionaries representing each masked section of the image.
        """
        
        return self.mask_generator.generate(image)
    
    # TODO
    def is_valid_masks(self, masks: list[dict]) -> bool:
        """
        Checks if the masks list is a valid list of masks.

        Arguments:
            masks (list[dict]): List of dictionaries representing each masked section of the image.

        Returns:
            bool: Whether or not masks is a valid list.
        """
        
        return True
        
    def prepare_mask(self, mask: dict, is_bw: bool = False) -> np.ndarray:
        """
        Gets the image array of the mask.

        Arguments:
            image (np.ndarray): Image represented in an array.
            mask (dict): Dictionary representing the mask to prepare.
            is_bw (bool) : Whether or not mask overlay has black and white filter.

        Returns:
            np.ndarray: Image array of the prepared mask.
        """

        img = np.ones((mask['segmentation'].shape[0], mask['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        
        m = mask['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        
        return img

    def prepare_masks(self, masks: list[dict], is_bw: bool = False) -> np.ndarray:
        """
        Gets the image array of a list of masks.

        Arguments:
            image (np.ndarray): Image represented in an array.
            masks (list[dict]): List of dictionaries representing each masked section of the image.
            is_bw (bool) : Whether or not mask overlay has black and white filter.

        Returns:
            np.ndarray: Image array of the prepared masks.
        """
        
        if len(masks) == 0:
            return
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

        img = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        
        for mask in sorted_masks:
            m = mask['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        
        return img
    
    def visualize(self, mask_overlay: np.ndarray, image: np.ndarray, figsize: tuple[int] = (10, 5)):
        """
        Visualizes the mask. Overlays it on the image (if any).

        Arguments:
            mask_overlay (np.ndarray): Mask overlay image array.
            image (np.ndarray): Image represented in an array.
            figsize (tuple[int]): Matplotlib figure size.

        """
        
        is_bw = len(mask_overlay.shape) == 2
        
        if is_bw:
            # Convert the mask to have 3 channels
            filter = np.stack([mask_overlay, mask_overlay, mask_overlay], axis=-1)

            # Apply the mask to the image
            masked_image = image * filter

            # Display the original image and the masked image
            plt.figure(figsize=figsize)
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(masked_image)
            plt.title('Masked Image')
            plt.axis('off')

            plt.show()
        else:
            # Display the original image and the masked image
            plt.figure(figsize=figsize)
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(image)
            ax = plt.gca()
            ax.set_autoscale_on(False)
            ax.imshow(mask_overlay)
            plt.title('Masked Image')
            plt.axis('off')
            
            plt.show()

    def convert_bw(self, mask_overlay: np.ndarray) -> np.ndarray:
        """
        Generate black and white filter on maks. Make any pixel not in overlay black
        and everything else white.

        Arguments:
            mask_overlay (np.ndarray) : Mask overlay image array

        Returns:
            np.ndarray : Black and white mask overlay
        """

        bw_overlay = np.all(mask_overlay[:, :, :3] == 1, axis=-1).astype(float)
        bw_overlay = np.where(bw_overlay==1., 0., 1.)

        return bw_overlay