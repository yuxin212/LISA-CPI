#!/bin/bash

set -e

if ! command -v wget &> /dev/null ; then
    echo "wget not found. Please install wget."
    exit 1
fi

echo "Downloading raw data..."
mkdir -p data/original
wget -O data/original/pain_raw.csv https://zenodo.org/record/8150913/files/pain_raw.csv?download=1
wget -O data/original/top20_raw.csv https://zenodo.org/record/8150913/files/top20_raw.csv?download=1
wget -O data/original/fda_raw.csv https://zenodo.org/record/8150913/files/fda_raw.csv?download=1
wget -O data/original/gut_raw.csv https://zenodo.org/record/8150913/files/gut_raw.csv?download=1
wget -O fastas.zip https://zenodo.org/record/8150913/files/fastas.zip?download=1
unzip fastas.zip -d alphafold/fastas/
rm fastas.zip

echo "All raw data downloaded."