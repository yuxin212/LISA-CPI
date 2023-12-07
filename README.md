# LISA-CPI
Official repository for ***A deep learning framework combining molecular image and protein structural representation identifies candidate drugs for chronic pain***

## Initial Setup

This code has been tested on Python versions from 3.7 to 3.10 and PyTorch versions from 1.12 to the current latest version (2.1.1). Any Python version newer than 3.11 or older than 3.6 is not supported. Please adjust the installation commands according to your machine and preferences. 

Install Anaconda/Miniconda, create a conda environment, and activate the conda environment

```bash
conda create -n lisa-cpi python=3.10 -y \
    && conda activate lisa-cpi
```

Install packages: 

if you have nVidia GPUs that support CUDA on your machine:

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
python -m pip install -r requirements.txt
```

otherwise:

```bash
conda install pytorch torchvision cpuonly -c pytorch -y
python -m pip install -r requirements.txt
```

Clone the repo to your machine

```bash
git clone https://github.com/yuxin212/LISA-CPI.git
cd LISA-CPI
```

Install 

```bash
python -m pip install -e .
```

If you want to use AlphaFold2 to generate intermediate representations by your self: **Please note: AlphaFold2 can only run in Linux systems**

Clone the modified AlphaFold2 repo:

```bash
git clone https://github.com/yuxin212/alphafold.git
cd alphafold
```

Follow [First time setup](https://github.com/deepmind/alphafold#first-time-setup) and step 1 and step 2 of [Running AlphaFold](https://github.com/deepmind/alphafold#running-alphafold) to prepare AlphaFold2 dataset and build docker image. 

## Prepare the dataset

Our dataset consists of 2 parts: 1) Ligand images generated using RDKit and 2) GPCR representations generated using AlphaFold2. For each part, we provide both the processed data and raw data with scripts to process the raw data.

### Download the processed data

We provide a script to download all the data and final models used in our paper. `cd` into this repository and make sure you have `wget` installed. 

*   Download all 4 processed datasets: Top-20 GPCR dataset, pain-related GPCR dataset, FDA-approved drugs dataset, and 380 gut-metabolites dataset: 

    ```bash
    ./scripts/download_data.sh
    ```

*   If you wish to download only one or some of the datasets or final trained model, you can use one or more of the following scripts:

    ```bash
    ./scripts/download_top20.sh # download Top-20 GPCR dataset only
    ./scripts/download_pain.sh # download pain-related GPCR dataset only
    ./scripts/download_fda.sh # download FDA-approved drugs dataset only
    ./scripts/download_gut.sh # download 380 gut-metabolites dataset only
    ./scripts/download_models.sh # download final trained models only
    ```

The `download_data.sh` will also download needed GPCR representations generated using AlphaFold2 and the final models used in our paper. The final directory structure of the downloaded data should look like this:

```
data/
├── representations/        # GPCR representations
│   ├── pain/               # Pain-related GPCR representations
│   ├── top20/              # Top-20 GPCR representations
├── ligands/                
│   ├── top20/              # Top-20 GPCR dataset
│   │   ├── anno/           # annotations for training set and test set
│   │   ├── imgs/           # ligand images
│   ├── pain/               # Pain-related GPCR dataset
│   │   ├── anno/           # annotations for training set and test set
│   │   ├── imgs/           # ligand images
├── pred/                   
│   ├── fda/                # FDA-approved drugs dataset
│   ├── gut/                # 380 gut-metabolites dataset
trained_models/
├── train/              
│   ├── original_models/    # final models used in our paper
```

### Download and process raw dataset or customize dataset

If you wish to download and process the raw data by yourself, you can follow the instructions below.

1. Download the raw data:

    ```bash
    ./scripts/download_raw_data.sh
    ```

    All the raw ligand data will be downloaded to `data/original` and `fasta` files of GPCRs will be downloaded to `alphafold/fastas`. 

2. Generate GPCR representations:

    **Please note:**

    **1. If you already have original AlphaFold2 docker image built on your machine and you built another docker image using our modified AlphaFold2, you may want to specify the docker image name for the modified AlphaFold2 when running the following commands.**

    **2. Generating each representation using the original AlphaFold2 may take several hours based on your machine and the length of amino acid sequence of each GPCR, using our modified AlphaFold2 can drastically reduce the time consumption as it runs only one model and no recycling is performed.**

    Please specify `$ALPHAFOLD_DATA_DIR` and `$ALPHAFOLD_OUTPUT_DIR` as absolute paths.

    ```bash
    cd alphafold
    python docker/run_docker.py \
        --fasta_paths=fastas/GPCR1.fasta,fastas/GPCR2.fasta,... \
        --data_dir=$ALPHAFOLD_DATA_DIR \
        --output_dir=$ALPHAFOLD_OUTPUT_DIR \
        --max_template_date=2021-11-01
    
    python extract_representations.py \
        --data-dir $ALPHAFOLD_OUTPUT_DIR \
        --out-dir ../data/representations 
    ```

2. Process the raw data to generate ligand images and datasets, here we use Top-20 GPCR dataset as an example:

    ```bash
    cd .. # go back to the root directory of this repo

    export DATASET=top20 # specify the dataset name
    export GPCR_COL=uniprot_id # specify the column name of GPCR in the raw data
    export SMILES_COL=smiles # specify the column name of SMILES in the raw data
    export LABEL_COL=pKi # specify the column name of label in the raw data
    export FILE_NAME_COL=inchi_key # specify the column name of file name in the raw data

    python scripts/prepare_dataset.py \
        --dataset data/original/${DATASET}_raw.csv \
        --gpcr-col ${GPCR_COL} \
        --smiles-col ${SMILES_COL} \
        --label-col ${LABEL_COL} \
        --file-name-col ${FILE_NAME_COL} \
        --rep-path data/representations/${DATASET}/{}.npy \
        --save-path data/ligands/${DATASET}/imgs \
        --anno-path data/ligands/${DATASET}/anno \
        -j 12 \
        --test-size 0.3 \
        --task regression
    ```

    After running the above command, annotations for training set and test set will be dumped in `data/ligands/${DATASET}/anno` and ligand images will be saved in `data/ligands/${DATASET}/imgs`.

## Running model

### Running model for inference

```bash
python scripts/prediction.py \
    --cfg configs/prediction/pain.yml \
    --data-dir data/pred/fda \
    --rep-path data/representations/pain/P08908.npy \
    --out-dir output/prediction/fda/P08908
```

### Inference output

The inference will output one file `per_sample_result.json` in the `--out-dir` directory. The file contains the predicted values for each sample in the dataset:

```json
[
    {
        "drug": "data/pred/fda/RUVINXPYWBROJD-ONEGZZNKSA-N.png",
        "protein": "data/representations/P08908.npy",
        "prediction": 1.234,
    },
    {
        "drug": "data/pred/fda/GJSURZIOUXUGAL-UHFFFAOYSA-N.png",
        "protein": "data/representations/P08908.npy",
        "prediction": 5.678,
    },
    ...
]
```

### Running model for training

Running the following command for training and evaluating the model without cross-validation:

If you generate dataset from donwloaded raw data, please specify annotation files for training set and test set in the config file accordingly. 

```bash
# training
python scripts/train.py \
    --cfg configs/train/${DATASET}.yml 

# evaluating
python scripts/train.py \
    --cfg configs/eval/${DATASET}.yml
```

Running the following command for training and evaluating the model with cross-validation:

```bash
# training
python scripts/kfold_cv.py \
    --cfg configs/train/<DATASET>_10fold.yml 

# evaluating
python scripts/kfold_cv.py \
    --cfg configs/eval/<DATASET>_10fold.yml
```

Running the following command for training and evaluating the model without using GPCR representations:

```bash
# training
python scripts/finetune.py \
    --cfg configs/train/<DATASET>_no_rep.yml 

# evaluating
python scripts/finetune.py \
    --cfg configs/eval/<DATASET>_no_rep.yml
```

### Training output

The training output will be saved in `trained_models/train` directory. `logs` folder contains the tensorboard logs of train, validation and test results. If you are using cross-validation, the trained models will be saved in `fold_0`, `fold_1`, ..., `fold_9` folders. If you are not using cross-validation. 
