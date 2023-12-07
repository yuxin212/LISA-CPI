#!/bin/bash

set -e

if ! command -v wget &> /dev/null ; then
    echo "wget not found. Please install wget."
    exit 1
fi

mkdir -p data

echo "Downloading top20 dataset..."
wget -O top20.zip https://zenodo.org/record/8150913/files/top20.zip?download=1
unzip top20.zip -d data/
rm top20.zip

echo "top20 data downloaded."