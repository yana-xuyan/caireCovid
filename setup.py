import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="caire-covid",
    version="0.0.1",
    author="yana",
    author_email="yxucb1229@gmail.com",
    description="system for covid-19.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yana-xuyan/caire-covid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
