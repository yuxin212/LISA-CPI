import os
import glob
import json
from datetime import datetime
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.model_selection import train_test_split
from functools import partial
from multiprocessing import Pool

def smiles2img(row, save_path, smiles_col, file_name_col):
    
    row = row[1]
    smile = row[smiles_col]
    file_name = row[file_name_col]
    mol = Chem.MolFromSmiles(smile)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224), returnPNG=False)
    save_path = os.path.join(save_path, '{}.png'.format(file_name))
    img.save(save_path)

def process_with_split_gpcr(data, args):

    whole_train_set = []
    whole_test_set = []

    now = datetime.now()
    cur_dt = now.strftime('%m%d%Y%H%M%S')

    gpcr_names = list(data[args.gpcr_col].drop_duplicates())
    for gpcr_idx, gpcr in enumerate(gpcr_names):
        print('Generating images for {}...'.format(gpcr))

        gpcr_data = data.loc[data[args.gpcr_col] == gpcr]
        gpcr_save_path = os.path.join(args.save_path, gpcr)
        gpcr_anno_path = os.path.join(args.anno_path, gpcr)
        func = partial(smiles2img, save_path=gpcr_save_path, smiles_col=args.smiles_col, file_name_col=args.file_name_col)

        if args.rep_path:
            rep = args.rep_path.format(gpcr)
        else:
            rep = None

        if not os.path.exists(gpcr_save_path):
            os.makedirs(gpcr_save_path)
        if not os.path.exists(gpcr_anno_path):
            os.makedirs(gpcr_anno_path)
        
        if args.workers > 1:
            pools = Pool(args.workers)
            pools.map(func, gpcr_data.iterrows())
            pools.close()
            pools.join()
        
        else:
            for idx, row in gpcr_data.iterrows():
                func(row)

        print('{} image data generated.'.format(gpcr))
                    
        train_set = []
        test_set = []

        X = list(gpcr_data[args.file_name_col].apply(lambda x: os.path.join(gpcr_save_path, '{}.png'.format(x))))
        y = gpcr_data[args.label_col].tolist()

        if args.label_col is None:
            print('No label column provided, exiting...')
            if gpcr_idx == len(gpcr_names) - 1:
                return
            else:
                continue
        elif args.test_size == 1.:
            y = list(gpcr_data[args.label_col])
            X_train, X_test, y_train, y_test = [], X, [], y
            for i in range(len(X_test)):
                label = y_test[i]
                test_set.append({
                    'data': X_test[i],
                    'label': label, 
                    'rep': rep
                })
                whole_test_set.append({
                    'data': X_test[i],
                    'label': label, 
                    'rep': rep
                })
        else:
            print('Generating dataset annotations for {}...'.format(gpcr))
            if args.task == 'regression':
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=args.test_size,
                    random_state=args.random_state
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=args.test_size,
                    random_state=args.random_state,
                    stratify=y
                )


            for i in range(len(X_train)):
                train_set.append({
                    'data': X_train[i],
                    'label': y_train[i], 
                    'rep': rep
                })
                whole_train_set.append({
                    'data': X_train[i],
                    'label': y_train[i], 
                    'rep': rep
                })
            for i in range(len(X_test)):
                test_set.append({
                    'data': X_test[i],
                    'label': y_test[i], 
                    'rep': rep
                })
                whole_test_set.append({
                    'data': X_test[i],
                    'label': y_test[i], 
                    'rep': rep
                })

        print('Saving annotations for {}...'.format(gpcr))
            
        train_anno = os.path.join(gpcr_anno_path, 'GPCR_train_{}.json'.format(cur_dt))
        test_anno = os.path.join(gpcr_anno_path, 'GPCR_test_{}.json'.format(cur_dt))

        with open(train_anno, 'w') as f:
            json.dump(train_set, f, indent=1)
            
        with open(test_anno, 'w') as f:
            json.dump(test_set, f, indent=1)
            
        print('Annotations for {} have been saved to {} and {}'.format(gpcr, train_anno, test_anno))
    
    whole_train_anno = os.path.join(args.anno_path, 'GPCR_train_{}.json'.format(cur_dt))
    whole_test_anno = os.path.join(args.anno_path, 'GPCR_test_{}.json'.format(cur_dt))

    with open(whole_train_anno, 'w') as f:
        json.dump(whole_train_set, f, indent=1)
        
    with open(whole_test_anno, 'w') as f:
        json.dump(whole_test_set, f, indent=1)
        
    print('Annotations for whole dataset have been saved to {} and {}'.format(whole_train_anno, whole_test_anno))

def process_without_split_gpcr(data, args):

    """Process the provided table dataset without splitting 
    """

    now = datetime.now()
    cur_dt = now.strftime('%m%d%Y%H%M%S')

    func = partial(smiles2img, save_path=args.save_path, smiles_col=args.smiles_col, file_name_col=args.file_name_col)

    if args.rep_path:
        if '*' in args.rep_path:
            rep = glob.glob(args.rep_path)
        else:
            rep = args.rep_path
    else:
        rep = None
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.anno_path):
        os.makedirs(args.anno_path)
    
    if args.workers > 1:
        pools = Pool(args.workers)
        pools.map(func, data.iterrows())
        pools.close()
        pools.join()
    else:
        for idx, row in data.iterrows():
            func(row)

    print('Image data generated.')
    
    train_set = []
    test_set = []
    X = list(data[args.file_name_col].apply(lambda x: os.path.join(args.save_path, '{}.png'.format(x))))
    y = data[args.label_col].tolist()

    if args.label_col is None:
        print('No label column provided, exiting...')
        return
    elif args.test_size == 1.:
        y = list(data[args.label_col])
        X_train, X_test, y_train, y_test = [], X, [], y
        for i in range(len(X_test)):
            label = y_test[i]
            if isinstance(rep, list):
                for gpcr_rep in rep:
                    test_set.append({
                        'data': X_test[i],
                        'label': label, 
                        'rep': gpcr_rep
                    })
            else:
                test_set.append({
                    'data': X_test[i],
                    'label': label, 
                    'rep': rep
                })
    else:
        print('Generating dataset annotations...')
        if args.task == 'regression':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=args.test_size,
                random_state=args.random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=args.test_size,
                random_state=args.random_state,
                stratify=y
            )


        for i in range(len(X_train)):
            if isinstance(rep, list):
                for gpcr_rep in rep:
                    train_set.append({
                        'data': X_train[i],
                        'label': y_train[i], 
                        'rep': gpcr_rep
                    })
            else:
                train_set.append({
                    'data': X_train[i],
                    'label': y_train[i], 
                    'rep': rep
                })
        for i in range(len(X_test)):
            if isinstance(rep, list):
                for gpcr_rep in rep:
                    test_set.append({
                        'data': X_test[i],
                        'label': y_test[i], 
                        'rep': gpcr_rep
                    })
            else:
                test_set.append({
                    'data': X_test[i],
                    'label': y_test[i], 
                    'rep': rep
                })
    
    print('Saving annotations for dataset...')

    train_anno = os.path.join(args.anno_path, 'train_{}.json'.format(cur_dt))
    test_anno = os.path.join(args.anno_path, 'test_{}.json'.format(cur_dt))

    with open(train_anno, 'w') as f:
        json.dump(train_set, f, indent=1)
    
    with open(test_anno, 'w') as f:
        json.dump(test_set, f, indent=1)
    
    print('Annotations for dataset have been saved to {} and {}'.format(train_anno, test_anno))

def main(args):
    
    print(args)
    
    assert (args.task in ['regression', 'classification']), 'Task must be either regression or classification'
    
    print('Reading data file...')
    if args.dataset.endswith('.xls') or args.dataset.endswith('.xlsx'):
        data = pd.read_excel(args.dataset, sheet_name=args.sheet_name)
    elif args.dataset.endswith('.csv'):
        data = pd.read_csv(args.dataset)
    else:
        raise Exception('Unsupported dataset file format')
    
    if args.file_name_col == None:
        args.file_name_col = 'index'
        data[args.file_name_col] = data.index

    if args.gpcr_col:
        process_with_split_gpcr(data, args)
    else:
        process_without_split_gpcr(data, args)
            
def argparser():
    
    import argparse
    
    parser = argparse.ArgumentParser(description='GPCR pre-training data preprocessing')
    parser.add_argument('--dataset', default='data/original/GPCR_Ligand_Paires_111301_top10_new_active37810.xlsx', help='original dataset file')
    parser.add_argument('--sheet-name', default=0, help='name of the sheet including data')
    parser.add_argument('--gpcr-col', default=None, help='name of the column containing gpcr names. \
                        if None, annotations will not be splitted by gpcr names')
    parser.add_argument('--smiles-col', required=True, help='name of the column containing smiles. ')
    parser.add_argument('--label-col', default=None, help='name of the column containing labels/targets. \
                        if None, no annotation files will be generated and dataset will be used for predictions only')
    parser.add_argument('--file-name-col', default=None, help='name of the column containing file names for to be saved molecular image files. \
                        if None, use original index as file names')
    parser.add_argument('--rep-path', default=None, help='path to AlphaFold representation files. e.g. data/rep/{}.npy or data/rep/*.npy in the case you want to test the drug on all available GPCRs or data/rep/P08908.npy \
                        if None, model will run without representations. ')
    parser.add_argument('--save-path', default='data/finetune/top-10/imgs/', help='path to save molecule images')
    parser.add_argument('--anno-path', default='data/finetune/top-10/anno/', help='path to save annotations')
    parser.add_argument('-j', '--workers', default=12, type=int, help='number of cpu to generate images')
    parser.add_argument('--test-size', default=0.3, type=float, help='ratio of test data set size')
    parser.add_argument('--random-state', default=None, type=int, help='random shuffle state applied to split data')
    parser.add_argument('--task', default='regression', help='whether perform regression or classification')
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    
    args = argparser()
    main(args)
