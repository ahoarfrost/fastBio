import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastBio",
    version="0.1.7",
    author="Adrienne Hoarfrost",
    author_email="adrienne.l.hoarfrost@gmail.com",
    description="Deep learning for biological sequences with fastai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahoarfrost/fastBio",
    packages=setuptools.find_packages(),
    install_requires = ['fastai==1.0.52', 'biopython==1.73', 'torch==1.2.0', 'torchvision==0.4.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
