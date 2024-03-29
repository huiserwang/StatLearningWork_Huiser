import numpy as np
import os
from knn import libKNN
from bayes import libBayes
from adaboost import libAdaboost
from convnet import libCNN
from svm import libSVM
import glob
import argparse

def op_load_csv(path):
    f = open(path, 'r')
    content = f.readlines()
    content = [x.strip() for x in content]
    content = content[1:]
    data = [(x.split(',')[0], int(x.split(',')[1])) for x in content]
    return data
def op_load_npy(path):
    data = np.load(path)
    return data
def op_write_csv(test_files, test_pred_labels, out):
    all_out = []
    all_out.append('id,category')
    for i,each in enumerate(test_files):
        all_out.append('{},{}'.format(each,int(test_pred_labels[i])))
    content = '\n'.join(all_out)
    f = open(out, 'w+')
    f.writelines(content)
    f.close()
def op_merge_data(labels, prefix, concat=True):
    """merge data

    Args:
        labels (list(tuple)): labels for each file in csv
        prefix (str): where to load npy
        concat (bool, optional): if concat all data along dimension 0 into one array. Defaults to True.

    Returns:
        label_data (np.ndarray | list(np.ndarray)): labels for all frames, shape [1334*100,] if concat=True
        feats_data (np.ndarray | list(np.ndarray)): feats for all frames, shape [1334*100, 15] if concat=True

    """
    label_data = []
    feats_data = []
    for each in labels:
        npy_data = op_load_npy(os.path.join(prefix, each[0]))
        label_data.append(np.zeros(npy_data.shape[0])+each[1])
        feats_data.append(npy_data)
    if concat:
        label_data = np.concatenate(label_data)
        feats_data = np.concatenate(feats_data)
    return label_data, feats_data
def op_get_test(path, concat=True):
    files = glob.glob(os.path.join(path, "*.npy"))
    names_list = []
    feats_list = []
    for each in files:
        npy_data = op_load_npy(each)
        feats_list.append(npy_data)
        names_list.append(each.split('/')[-1])
    if concat:
        test_data = np.concatenate(feats_list)
    return names_list, test_data

def preprocess_data(feats, label=None, mode='norm', shuffle=True, test=False):
    """preprocess func

    Args:
        mode (str, optional): what you want to do. Defaults to 'norm'.

    Returns:
        label_data (np.ndarray): labels for all frames, shape [1334*100,]
        feats_data (np.ndarray): feats for all frames, shape [1334*100, 15]
    """
    if mode == 'norm':
        N = feats.shape[0] if isinstance(feats, np.ndarray) else len(feats)*feats[0].shape[0]
        if label is not None:
            label_data = label
        if test:
            feats_data = (feats - feats.min(axis=1).reshape(N,1)) / (feats.max(axis=1) - feats.min(axis=1) + 1e-5).reshape(N,1)
        else:
            feats_data = (feats - feats.min(axis=1).reshape(N,1)) / (feats.max(axis=1) - feats.min(axis=1)).reshape(N,1)
            #feats_data = feats
            unkeep = np.unique(np.argwhere(np.isnan(feats_data)==True)[:,0])
        keep = np.ones(feats_data.shape[0]).astype(np.bool)
        if not test:
            keep[unkeep] = False
    elif mode == 'unnorm':
        pass
    idx = np.arange(feats_data[keep].shape[0])
    if shuffle:
        np.random.shuffle(idx)
    if label is not None:
        return label_data[keep][idx], feats_data[keep][idx]
    else:
        return feats_data[keep][idx]

def main(args):
    train_dir = './train/'
    test_dir  = './test/'
    label_train = './label_train.csv'
    out_path = args.out

    # load labels and corresponding feats
    label = op_load_csv(label_train)
    labels,feats = op_merge_data(label, train_dir, concat=True)
    test_names, test_feat = op_get_test(test_dir)

    # preprocess
    train_labels, train_feats = preprocess_data(feats, label=labels, mode='norm', shuffle=False) #133400 frames, each frame has 15 channels.  #shuffle=False for cnn
    test_feats = preprocess_data(test_feat, mode='norm', shuffle=False, test=True)

    # model init and training
    #KNN model
    if args.method == 'knn':
        classifier = libKNN(n_neighbors=5, algo='auto')
    #SVM model
    elif args.method == 'svm':
        classifier = libSVM(C=1.0, kernel='rbf',gamma=0.1)
    #Bayes model
    elif args.method == 'bayes':
        classifier = libBayes(n=20)
    #AdaBoost
    elif args.method == 'adaboost':
        classifier = libAdaboost(
                        base_estimator_cfg={"max_depth":3, "min_samples_leaf":5, "min_samples_split":20},
                        n_estimators=600,
                        lr=0.8)
    #deep model
    elif args.method == 'cnn':
        classifier = libCNN(train_dir, test_dir, label_train, epoch=500)
    else:
        ValueError("Only SVM, KNN, AdaBoost, Bayes Classifier and ConvNet are available.")

    #model training
    if args.method == 'cnn':
        classifier.train()  #for cnn
        
    else:
        classifier.train(train_labels, train_feats)  #for non cnn

    # evaluation on training set
    if args.method == 'cnn':
        print('eval has been done in training!')
    else:
        classifier.eval(train_labels, train_feats)

    # inference for test set
    if args.method == 'cnn':
        pred_labels = classifier.test()  #for cnn
    else:
        pred_labels = classifier.test(test_feats)
    
    # test_file_names: all file names, [0023.npy, 1245.pny, ....]
    # test_pred_labels: predicted labels for each file, [0,1,2,0,....]
    # len(test_file_names) == len(test_pred_labels)
    test_file_names = test_names
    test_pred_labels = pred_labels
    op_write_csv(test_file_names, test_pred_labels, out=out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='knn')
    parser.add_argument('--out', type=str, default='/home/huiser/Desktop/Codes/statLearningWork_SJTU/default_exp_knn.csv')
    args = parser.parse_args()
    main(args)