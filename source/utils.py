import os
from nltk import word_tokenize, sent_tokenize, pos_tag

train_data_dir = '/Users/qbuser/Documents/pythonWorks/BigDataWorks/qint_nlp/source/data/training_resource'


def is_txt_file(filename):
    return filename.endswith('.txt')


def training_testing_dataset(self):
    tagged_sents = []
    try:
        files = [f for f in os.listdir(train_data_dir) if os.path.isfile(
            os.path.join(train_data_dir, f))]
    except Exception as e:
        raise e

    for file in files:
        file_path = os.path.join(train_data_dir, file)
        if is_txt_file(file_path):
            with open(file_path, 'r') as f:
                sents = sent_tokenize(f.read())
                for sent in sents:
                    tagged_sents.append(pos_tag(word_tokenize(sent)))

    size = int(len(tagged_sents) * 0.9)
    train_sents = tagged_sents[:size]
    test_sents = tagged_sents[size:]
    return train_sents, test_sents
