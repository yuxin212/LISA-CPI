#!/bin/bash

set -e

if ! command -v wget &> /dev/null ; then
    echo "wget not found. Please install wget."
    exit 1
fi

mkdir -p data

echo "Downloading fda approved dataset..."
wget -O fda.zip https://zenodo.org/record/8150913/files/fda.zip?download=1
unzip fda.zip -d data/
rm fda.zip

echo "fda approved data downloaded."