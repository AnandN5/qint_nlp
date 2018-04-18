from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import os
import utils
from nltk import word_tokenize, sent_tokenize, pos_tag


train_data_dir = '/Users/qbuser/Documents/pythonWorks/BigDataWorks/qint_nlp/source/data/training_resource'
clf = Pipeline([('vectorizer', DictVectorizer(sparse=False)),
                ('classifier', DecisionTreeClassifier(criterion='entropy'))])


class CustomTrainedTagger(object):
    """Class to train pos tag in custom way using DecisionTreeClassifier."""

    def __init__(self):
        train_sents, test_sents = self.__training_testing_dataset()
        self.__text_classifier(train_sents, test_sents)

    def tag(self, sentences):
        sentences = sent_tokenize(sentences)
        tagged_sentences = []

        import pickle
        with open('customtagger.pkl', 'rb') as inp:
            clf = pickle.load(inp)
            for sent in sentences:
                sent = word_tokenize(sent)
                tags = clf.predict([self.__features(sent, index)
                                    for index in range(len(sent))])
                zipped = zip(sent, tags)
                tagged_sentences.append([x for x in zipped])
        return tagged_sentences

    def __training_testing_dataset(self):
        tagged_sents = []

        try:
            files = [f for f in os.listdir(train_data_dir) if os.path.isfile(
                os.path.join(train_data_dir, f))]
        except Exception as e:
            raise e

        for file in files:
            file_path = os.path.join(train_data_dir, file)
            if utils.is_txt_file(file_path):
                with open(file_path, 'r') as f:
                    sents = sent_tokenize(f.read())
                    for sent in sents:
                        tagged_sents.append(pos_tag(word_tokenize(sent)))

        size = int(len(tagged_sents) * 0.9)
        train_sents = tagged_sents[:size]
        test_sents = tagged_sents[size:]
        return train_sents, test_sents

    def __features(self, tagged_sents, index):
        """ tagged sentence: [w1, w2, ...], index: the index of the word """
        return {
            'word': tagged_sents[index],
            'is_first': index == 0,
            'is_last': index == len(tagged_sents) - 1,
            'is_capitalized': tagged_sents[index][0].upper() == tagged_sents[index][0],
            'is_all_caps': tagged_sents[index].upper() == tagged_sents[index],
            'is_all_lower': tagged_sents[index].lower() == tagged_sents[index],
            'prefix-1': tagged_sents[index][0],
            'prefix-2': tagged_sents[index][:2],
            'prefix-3': tagged_sents[index][:3],
            'suffix-1': tagged_sents[index][-1],
            'suffix-2': tagged_sents[index][-2:],
            'suffix-3': tagged_sents[index][-3:],
            'prev_word': '' if index == 0 else tagged_sents[index - 1],
            'next_word': '' if index == len(tagged_sents) - 1 else tagged_sents[index + 1],
            'has_hyphen': '-' in tagged_sents[index],
            'is_numeric': tagged_sents[index].isdigit(),
            'capitals_inside': tagged_sents[index][1:].lower() != tagged_sents[index][1:]
        }

    def __text_classifier(self, train_sents, test_sents):
        X, y = self.__transform_to_dataset(train_sents)
        # Use only the first 10K samples if you're running it multiple times.
        clf.fit(X[:10000], y[:10000])
        print('Training completed')

        X_test, y_test = self.__transform_to_dataset(test_sents)
        print("Accuracy:", clf.score(X_test, y_test))

        import pickle
        with open('customtagger.pkl', 'wb') as outp:
            pickle.dump(clf, outp, -1)

    def __untag_sentence(self, tagged_sentences):
        return [w for w, t in tagged_sentences]

    def __transform_to_dataset(self, tagged_sentences):
        X, y = [], []
        for tagged in tagged_sentences:
            for index in range(len(tagged)):
                X.append(self.__features(self.__untag_sentence(tagged), index))
                y.append(tagged[index][1])
        return X, y
