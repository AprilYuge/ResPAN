import setuptools

with open("./README.md", "r", encoding='utf-8') as fh:
  long_description = fh.read()

setuptools.setup(
    name="ResPAN",
    version="0.1.0",
    keywords=['scRNA-seq', 'GAN', 'batch effect correction', 'neural network']
    author="Yuge Wang, Tianyu Liu",
    author_email="wangyuge22@qq.com",
    description="A light structured residual autoencoder and mutual nearest neighbor paring guided adversarial network for scRNA-seq batch correction. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT License'
    url="https://github.com/AprilYuge/ResPAN",
    install_requires = [
    'numpy',
    'pandas',
    'scipy',
    'torch',
    'sklearn',
    'scanpy'
    ],
    packages=['ResPAN'],
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
)