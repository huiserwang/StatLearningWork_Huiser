from sklearn.ensemble import AdaBoostClassifier as ABclassifier
from sklearn.tree import DecisionTreeClassifier as DTclassifier
import numpy as np
class libAdaboost():
    def __init__(self, base_estimator_cfg, n_estimators=200, lr=0.8, algo='SAMME'):
        self.base_cfg=base_estimator_cfg
        self.n_estimators=n_estimators
        self.lr = lr
        self.algo = algo
        self.adaboost = ABclassifier(
                                    DTclassifier(
                                        max_depth=self.base_cfg['max_depth'],
                                        min_samples_leaf=self.base_cfg['min_samples_leaf'],
                                        min_samples_split=self.base_cfg['min_samples_split']
                                    ),
                                    algorithm=self.algo,
                                    n_estimators=self.n_estimators,
                                    learning_rate=self.lr)

    def train(self, train_labels, train_feats):
        self.adaboost.fit(train_feats, train_labels)
        print('train done')

    def eval(self, train_labels, train_feats):
        pred = self.adaboost.predict(train_feats)
        diff = pred - train_labels
        correctN = np.argwhere(diff == 0.).shape[0]
        print('{}'.format(correctN/len(train_labels)))
        
    def test(self, test_feats):
        pred_label = []
        pred = self.adaboost.predict(test_feats)
        for i in range(0,len(pred),100):
            count = np.bincount(pred[i:i+100].astype(np.int64))
            pred_label.append(np.argmax(count))
        print('test done')
        return pred_label