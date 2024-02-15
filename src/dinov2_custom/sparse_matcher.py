# src/dinov2_custom/sparse_matcher.py

from .dinov2 import DINOv2

from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_closing, binary_opening

class Sparse_Matcher:
    def __init__(self):
        pass
    
    def match_features(self, features1: np.ndarray, features2: np.ndarray) -> (np.ndarray, np.ndarray):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(features1)
        distances, match2to1 = knn.kneighbors(features2)
        match2to1 = np.array(match2to1)
        
        return distances, match2to1

    def visualize(self, 
                  dinov2: DINOv2,
                  image1: np.ndarray, 
                  image2: np.ndarray, 
                  mask1: np.ndarray, 
                  mask2: np.ndarray, 
                  grid_size1: tuple[int], 
                  grid_size2: tuple[int], 
                  resize_scale1: float,
                  resize_scale2: float,
                  distances: np.ndarray,
                  match2to1: np.ndarray,
                  figsize: tuple[int] = (20, 20),
                  show_percentage: float = 1):
        
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(image1)
        ax1.axis("off")
        ax2.imshow(image2)
        ax2.axis("off")

        for idx2, (dist, idx1) in enumerate(zip(distances, match2to1)):
            row, col = dinov2.idx_to_source_position(idx1, grid_size1, resize_scale1)
            xyA = (col, row)
            if not mask1[int(row), int(col)]: continue # skip if feature is not on the object

            row, col = dinov2.idx_to_source_position(idx2, grid_size2, resize_scale2)
            xyB = (col, row)
            if not mask2[int(row), int(col)]: continue # skip if feature is not on the object

            if np.random.rand() > show_percentage: continue # sparsely draw so that we can see the lines...

            con = ConnectionPatch(xyA=xyB, xyB=xyA, coordsA="data", coordsB="data",
                                    axesA=ax2, axesB=ax1, color=np.random.rand(3,))
            ax2.add_artist(con)