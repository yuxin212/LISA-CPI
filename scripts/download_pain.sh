#!/bin/bash

set -e

if ! command -v wget &> /dev/null ; then
    echo "wget not found. Please install wget."
    exit 1
fi

mkdir -p data

echo "Downloading pain dataset..."
wget -O pain.zip https://zenodo.org/record/8150913/files/pain.zip?download=1
unzip pain.zip -d data/
rm pain.zip

echo "pain data downloaded."