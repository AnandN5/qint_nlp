from nltk import NaiveBayesClassifier

training_data_path = '/Users/qbuser/Documents/pythonWorks/BigDataWorks/qint_nlp/source/data/training_resource/'


class KeyRecognizer(object):
    def __init__(self):
        self.classifier = 'NaiveBayesClassifier'

    def train(self):
        audit_number_data_path = training_data_path + 'audit number.txt'
        non_conf_id_data_path = training_data_path + 'non conformace id.txt'
        paths = [(audit_number_data_path, 'audit number'),
                 (non_conf_id_data_path, 'non_conf_id')]
        all_data = []
        for path in paths:
            with open(path[0], 'r') as fin:
                data = [tuple(d.split()) for d in fin.read().split('\n')]
                all_data.extend([(x, path[1]) for x in data])
        print(all_data)
