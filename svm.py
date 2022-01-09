from sklearn import svm
import numpy as np

class libSVM():
    def __init__(self, C=1, kernel='rbf',gamma=0.1):
        self.clf = svm.SVC(C=C,kernel=kernel,gamma=gamma)

    def train(self, train_labels, train_feats):
        self.clf.fit(train_feats, train_labels)
        print('train done')

    def eval(self, train_labels, train_feats):
        pred = self.clf.predict(train_feats)
        diff = pred - train_labels
        correctN = np.argwhere(diff == 0.).shape[0]
        print('{}'.format(correctN/len(train_labels)))

    def test(self, test_feats):
        pred_label = []
        pred = self.clf.predict(test_feats)
        for i in range(0,len(pred),100):
            count = np.bincount(pred[i:i+100].astype(np.int64))
            pred_label.append(np.argmax(count))
        print('test done')
        return pred_label