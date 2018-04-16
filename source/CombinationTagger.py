import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import os
import utils

train_data_dir = '/Users/qbuser/Documents/pythonWorks/BigDataWorks/qint_nlp/source/data/training_resource'


class NGramTagger:

    def __init__(self):
        train_sents, test_sents = self.__training_testing_dataset()
        self.__train_tagger(self.__tagging_model(
            train_sents, test_sents))

    def tag(self, sentences):
        token_sentences = sent_tokenize(sentences)
        with open('b_tagger.pkl', 'rb') as tag_in:
            tagger = pickle.load(tag_in)
            return tagger.tag(
                [w for sent in token_sentences for w in word_tokenize(sent)])

    def __training_testing_dataset(self):
        try:
            files = [f for f in os.listdir(train_data_dir) if os.path.isfile(
                os.path.join(train_data_dir, f))]
        except Exception as e:
            raise e
        sentences = []
        tagged_sents = []
        for file in files:
            file_path = os.path.join(train_data_dir, file)
            if utils.is_txt_file(file_path):
                with open(file_path, 'r') as f:
                    sents = sent_tokenize(f.read())
                    sentences.extend(sents)
        for sent in sentences:
            tagged_sents.append(nltk.pos_tag(word_tokenize(sent)))
        size = int(len(tagged_sents) * 0.9)
        train_sents = tagged_sents[:size]
        test_sents = tagged_sents[size:]
        return train_sents, test_sents

    def __tagging_model(self, train_sents, test_sents):
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
        # default tagger
        default_tagger = nltk.DefaultTagger('NIL')

        # Unigram Tagger
        u_tagger = nltk.UnigramTagger(
            model=tag_model, cutoff=1, backoff=default_tagger)

        # Bigram_tagger
        b_tagger = nltk.BigramTagger(
            model=tag_model, cutoff=1, backoff=u_tagger)

        with open('b_tagger.pkl', 'wb') as out:
            pickle.dump(b_tagger, out, -1)
