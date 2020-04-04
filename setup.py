import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="caireCovid",
    version="0.0.1",
    author="Yan Xu",
    author_email="yxucb1229@gmail.com",
    description="system for covid-19.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yana-xuyan/caireCovid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "collections",
        "copy",
        "json",
        "six",
        # for downloading models over HTTPS
        "requests",
        # progress bars in model download and training scripts
        "tqdm >= 4.27",
        # for XLNet
        "sentencepiece",
    ],
    python_requires='>=3.6',
)
