# from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.pipeline import Pipeline
import utils
import nltk
from nltk import NaiveBayesClassifier
from Feature_extractor import word_tag_features
import os


# clf = Pipeline([('vectorizer', DictVectorizer(sparse=False)),
#                 ('classifier', DecisionTreeClassifier(criterion='entropy'))])


class CustomTrainedTagger(nltk.TaggerI):
    """
       Class to train pos tag in custom way using DecisionTreeClassifier.
       Context which considers while tagging takes previous word and its 'tag'
       i.e history of tags
    """

    def __init__(self):
        self.train_sents, self.test_sents = utils.training_testing_dataset()
        train_set = self.__transformed_train_set(self.train_sents)
        self.classifier = NaiveBayesClassifier.train(train_set)
        print('Trained tagger training completed')
        # print('Accuracy: ', self.evaluate(self.test_sents))
        import pickle
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_tagger.pkl'), 'wb') as out:
            clf = self.classifier
            pickle.dump(clf, out, -1)

    def tag(self, sent):
        history = []
        for i, word in enumerate(sent):
            featureset = word_tag_features(sent, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        zipped = zip(sent, history)
        tagged = [x for x in zipped]
        return tagged

    # def __features(self, sentence, i, history):
    #     features = {
    #         "suffix(1)": sentence[i][-1:],
    #         "suffix(2)": sentence[i][-2:],
    #         "suffix(3)": sentence[i][-3:]
    #     }
    #     if i == 0:
    #         features["prev-word"] = "<START>"
    #         features["prev-tag"] = "<START>"
    #     else:
    #         features["prev-word"] = sentence[i - 1]
    #         features["prev-tag"] = history[i - 1]
    #     return features

    def __untag_sentence(self, tagged_sentences):
        return [w for w, t in tagged_sentences]

    def __transformed_train_set(self, tagged_sentences):
        train_set = []
        for tagged in tagged_sentences:
            history = []
            for index in range(len(tagged)):
                featureset = word_tag_features(
                    self.__untag_sentence(tagged), index, history)
                tag = tagged[index][1]
                train_set.append((featureset, tag))
                history.append(tag)
        return train_set
