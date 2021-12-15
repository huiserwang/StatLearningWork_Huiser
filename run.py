import numpy as np
import os

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
    f = open(out, 'w')
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

def preprocess_data(label, feats, mode='norm'):
    """preprocess func

    Args:
        mode (str, optional): what you want to do. Defaults to 'norm'.

    Returns:
        label_data (np.ndarray): labels for all frames, shape [1334*100,]
        feats_data (np.ndarray): feats for all frames, shape [1334*100, 15]
    """
    if mode == 'norm':
        N = label.shape[0] if isinstance(label, np.ndarray) else len(label)*label[0].shape[0]
        label_data = label
        feats_data = (feats - feats.min(axis=1).reshape(N,1)) / (feats.max(axis=1) - feats.min(axis=1)).reshape(N,1)
    elif mode == 'xxx':
        pass
    return label_data, feats_data

def main():
    train_dir = './train/'
    test_dir  = './test/'
    label_train = './label_train.csv'

    # load labels and corresponding feats
    label = op_load_csv(label_train)
    labels,feats = op_merge_data(label, train_dir, concat=True)
    # preprocess
    train_labels, train_feats = preprocess_data(labels, feats, mode='norm') #133400 frames, each frame has 15 channels.

    # model init and training

    # evaluation on training set

    # inference for test set
    # test_file_names: all file names, [0023.npy, 1245.pny, ....]
    # test_pred_labels: predicted labels for each file, [0,1,2,0,....]
    # len(test_file_names) == len(test_pred_labels)
    test_file_names = None
    test_pred_labels = None
    op_write_csv(test_file_names, test_pred_labels, out='~/test_results.csv')

if __name__ == "__main__":
    main()