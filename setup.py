# setup.py

from setuptools import setup, find_packages

setup(
    name="dinov2_custom",
    version="0.1.0",
    description="A Python package for Sparse Matching using Meta's DINOv2 and Semantic Segmentation with Meta's Segment Anything Model (SAM)",
    author="Firdavs Nasriddinov",
    author_email="firdavs@caltech.edu",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "matplotlib==3.7.1",
        "numpy==1.23.5",
        "onnx==1.15.0",
        "onnxruntime==1.16.3",
        "opencv-python==4.8.0.76",
        "Pillow==9.4.0",
        "pycocotools==2.0.7",
        "scikit-learn==1.2.2",
        "scipy==1.11.4",
        "segment-anything==1.0"
        # torch and torchvision are handled separately due to URL-based installation
    ],

    python_requires='>=3.8',
)