from nltk import NaiveBayesClassifier, TaggerI
from Feature_extractor import chunk_features
import utils


class NgramChunkTagger(TaggerI):
    def __init__(self):
        self.train_set, self.test_set = utils.chunked_training_dataset()
        print('Ngram chunk tagger training started')
        self.classifier = NaiveBayesClassifier.train(
            self.__transformed_training_set())
        print('Ngram chunk tagger training completed')

    def tag(self, tagged_sent):
        history = []
        tags = [t for (w, t) in tagged_sent]
        for i, (word, tag) in enumerate(tagged_sent):
            featureset = chunk_features(tags, i, history)
            iob_tag = self.classifier.classify(featureset)
            history.append(iob_tag)
        result = [x for x in zip(tagged_sent, history)]
        return [(x, y, z) for ((x, y), z) in result]

    def __transformed_training_set(self):
        history = []
        train_set = []
        for a_set in self.train_set:
            tags = [t for (t, c) in a_set]
            for index in range(len(a_set)):
                iob_tag = a_set[index][1]
                featureset = chunk_features(tags, index, history)
                history.append(iob_tag)
                train_set.append((featureset, iob_tag))
        return train_set
