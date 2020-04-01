# pylint: disable = C0111
from setuptools import setup

with open("README.md", "r") as f:
    DESCRIPTION = f.read()

setup(name="covid19_kaggle",
      version="1.0.0",
      author="HKUST",
      description="CORD-19 Analysis",
      long_description=DESCRIPTION,
      long_description_content_type="text/markdown",
      url="https://github.com/****",
      project_urls={
          "Documentation": "https://github.com/***",
          "Issue Tracker": "https://github.com/***/issues",
          "Source Code": "https://github.com/***",
      },
      license="MIT License: http://opensource.org/licenses/MIT",
      packages=["covid19_kaggle"],
      package_dir={"": "src/python/"},
      keywords="kaggle hkust toolbox",
      python_requires=">=3.6",
      entry_points={
          "console_scripts": [
              "covid19_kaggle = covid19_kaggle.shell:main",
          ],
      },
      install_requires=[
          "tqdm>=4.43.0",
          "numpy>=1.17.4",
      ],
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 3",
          "Topic :: Software Development",
          "Topic :: Text Processing :: Indexing",
          "Topic :: Utilities"
      ])