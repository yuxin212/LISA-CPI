#!/bin/bash

set -e

if ! command -v wget &> /dev/null ; then
    echo "wget not found. Please install wget."
    exit 1
fi

mkdir -p data

echo "Downloading gut dataset..."
wget -O gut.zip https://zenodo.org/record/8150913/files/gut.zip?download=1
unzip gut.zip -d data/
rm gut.zip

echo "gut data downloaded."