#!/bin/bash

set -e

#!/bin/bash

set -e

if ! command -v wget &> /dev/null ; then
    echo "wget not found. Please install wget."
    exit 1
fi

mkdir -p trained_models/train/original_models

echo "Downloading final models..."
wget -O trained_models/train/original_models/pain.pth https://zenodo.org/record/8150913/files/pain.pth?download=1
wget -O trained_models/train/original_models/top20.pth https://zenodo.org/record/8150913/files/top20.pth?download=1
wget -O trained_models/train/original_models/imagemol.pth https://zenodo.org/record/8150913/files/imagemol.pth?download=1

echo "Final models downloaded."