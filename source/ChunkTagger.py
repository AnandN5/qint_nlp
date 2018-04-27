from nltk import NaiveBayesClassifier, TaggerI
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
        print(tagged_sent[0])
        tags = [t for (w, t) in tagged_sent]
        for i, (word, tag) in enumerate(tagged_sent):
            featureset = self.__features(tags, i, history)
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
                featureset = self.__features(tags, index, history)
                history.append(iob_tag)
                train_set.append((featureset, iob_tag))
        return train_set

    def __features(self, sent_tags, i, history):
        tag = sent_tags[i]
        if i == 0:
            prevtag = "<START>"
        else:
            prevtag = sent_tags[i - 1]
        if i == len(sent_tags) - 1:
            nexttag = "<END>"
        else:
            nexttag = sent_tags[i + 1]

        features = {
            'tag': tag,
            'is-first': True if i == 0 else False,
            'is-last': True if i == len(sent_tags) - 1 else False,
            'tag-1': '' if i == 0 else sent_tags[i - 1],
            'tag-2': '' if i <= 1 else sent_tags[i - 2],
            'iob-1': '' if i == 0 else history[i - 1],
            'iob-2': '' if i <= 1 else history[i - 2],
            'tag+1': '' if i == len(sent_tags) - 1 else sent_tags[i + 1],
            'tag+2': '' if i >= len(sent_tags) - 2 else sent_tags[i + 2],
            'prevtag+tag': '%s+%s' % (prevtag, tag),
            'tag+nexttag': '%s+%s' % (tag, nexttag),
            'tags_since_dt': utils.tags_since_dt(sent_tags, i)
        }
        return features
