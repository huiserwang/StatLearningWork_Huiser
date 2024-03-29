from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
class libKNN():
    def __init__(self, n_neighbors=3, algo='auto'):
        self.n_neighbors=n_neighbors
        self.algo = algo
        self.knn = KNN(n_neighbors=self.n_neighbors, algorithm=self.algo)

    def train(self, train_labels, train_feats):
        self.knn.fit(train_feats, train_labels)
        print('train done')

    def eval(self, train_labels, train_feats):
        pred = self.knn.predict(train_feats)
        diff = pred - train_labels
        correctN = np.argwhere(diff == 0.).shape[0]
        print('{}'.format(correctN/len(train_labels)))
        
    def test(self, test_feats):
        pred_label = []
        pred = self.knn.predict(test_feats)
        for i in range(0,len(pred),100):
            count = np.bincount(pred[i:i+100].astype(np.int64))
            pred_label.append(np.argmax(count))
        print('test done')
        return pred_label

