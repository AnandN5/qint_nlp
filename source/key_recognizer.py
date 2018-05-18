from nltk import NaiveBayesClassifier
from Feature_extractor import key_feature
import nltk
import utils


class KeyRecognizer(object):
    def __init__(self):
        self.classifier = 'NaiveBayesClassifier'

    def train(self):
        train_data, testing_data = utils.key_training_testing_dataset()
        print('key training started')
        train_featureset = [(key_feature(data[0]), data[1])
                            for data in train_data]
        test_featureset = [(key_feature(data[0]), data[1])
                           for data in testing_data]
        self.classifier = NaiveBayesClassifier.train(train_featureset)
        print('training completed')
        print('Accuracy:', nltk.classify.util.accuracy(
            self.classifier, test_featureset))

    def tag(self, sent_tree):
        audits = []
        non_confs = []
        for subtree in list(sent_tree.subtrees()):
            for leaf in subtree.leaves():
                if leaf[1] == 'CD' or leaf[1] == 'JJ':
                    featureset = key_feature(leaf)
                    aClass = self.classifier.classify(featureset)
                    import pdb
                    pdb.set_trace()
                    if aClass == 'audit number':
                        audits.append(leaf)
                    elif aClass == 'non_conf_id':
                        non_confs.append(leaf)
        return audits, non_confs
