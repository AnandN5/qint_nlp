import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import utils

train_data_dir = '/Users/qbuser/Documents/pythonWorks/BigDataWorks/qint_nlp/source/data/training_resource'


class NGramTagger:

    def __init__(self):
        self.train_sents, self.test_sents = utils.training_testing_dataset()
        self.__train_tagger(self.__tagging_model())

    def tag(self, sentences):
        token_sentences = sent_tokenize(sentences)
        with open('b_tagger.pkl', 'rb') as tag_in:
            tagger = pickle.load(tag_in)
            return tagger.tag(
                [w for sent in token_sentences for w in word_tokenize(sent)])


    def __tagging_model(self):
        train_sents = self.train_sents
        train_token_list = []
        train_token_list.extend([w for sent in train_sents for w in sent])

        # take the most likely tag for freq words and create a model
        freq_words = nltk.FreqDist(
            [t[0] for t in train_token_list]).most_common()
        cfd_freq_words = nltk.ConditionalFreqDist(train_token_list)
        likely_tags = dict((word, cfd_freq_words[word].max())
                           for (word, _) in freq_words)
        return likely_tags

    def __train_tagger(self, tag_model):
        test_sents = self.test_sents
        # default tagger
        default_tagger = nltk.DefaultTagger('NIL')

        # Unigram Tagger
        u_tagger = nltk.UnigramTagger(
            model=tag_model, cutoff=1, backoff=default_tagger)

        # Bigram_tagger
        b_tagger = nltk.BigramTagger(
            model=tag_model, cutoff=1, backoff=u_tagger)

        print('Accuracy :', b_tagger.evaluate(test_sents))

        with open('b_tagger.pkl', 'wb') as out:
            pickle.dump(b_tagger, out, -1)
