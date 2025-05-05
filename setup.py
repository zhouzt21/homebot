from setuptools import setup, find_packages

setup(
    name="homebot_sapien",
    version="0.2.0",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy<1.24",
        "scipy",
        "gymnasium>=0.28.1",
        "sapien==2.2.2",
        "h5py",
        "pyyaml",
        "tqdm",
        "GitPython",
        "tabulate",
        "gdown>=4.6.0",
        "transforms3d",
        "opencv-python",
        "imageio",
        "imageio[ffmpeg]",
        "trimesh",
        "rtree",
        "torch==1.11.0+cu113",       # 如需安装 GPU 版本，请参考官方说明添加 --extra-index-url
        "torchvision==0.12.0+cu113",   # 同上
        "matplotlib",
        "pre-commit",
        "wandb",
        "huggingface-hub==0.17.3",
        "timm==0.9.7",
        "tokenizers==0.14.1",
        "transformers==4.34.0",
        "diffusers==0.11.1"
    ],
    author="zhouzt21",
    author_email="zhouzt6@gmail.com",
    description="Description of the homebot_sapien package",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)