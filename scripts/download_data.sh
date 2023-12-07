#!/bin/bash

set -e

if ! command -v wget &> /dev/null ; then
    echo "wget not found. Please install wget."
    exit 1
fi

mkdir -p data
mkdir -p trained_models/train/original_models

echo "Downloading top20 GPCR dataset..."
wget -O top20.zip https://zenodo.org/record/8150913/files/top20.zip?download=1
unzip -o top20.zip -d data/
rm top20.zip

echo "Downloading pain GPCR dataset..."
wget -O pain.zip https://zenodo.org/record/8150913/files/pain.zip?download=1
unzip -o pain.zip -d data/
rm pain.zip

echo "Downloading fda approved dataset..."
wget -O fda.zip https://zenodo.org/record/8150913/files/fda.zip?download=1
unzip -o fda.zip -d data/
rm fda.zip

echo "Downloading gut dataset..."
wget -O gut.zip https://zenodo.org/record/8150913/files/gut.zip?download=1
unzip -o gut.zip -d data/
rm gut.zip

echo "Downloading final models..."
wget -O trained_models/train/original_models/pain.pth https://zenodo.org/record/8150913/files/pain.pth?download=1
wget -O trained_models/train/original_models/top20.pth https://zenodo.org/record/8150913/files/top20.pth?download=1
wget -O trained_models/train/original_models/imagemol.pth https://zenodo.org/record/8150913/files/imagemol.pth?download=1

echo "All data downloaded."