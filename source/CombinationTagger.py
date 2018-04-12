import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk


class CombinationTagger:

    def __init__(self, sentences):
        self.sentences = sentences
        self.tagger = 'Bigram Tagger'
        self._main()

    def _main(self):
        sentences = sent_tokenize(self.sentences)
        tagged_sents = []
        for sent in sentences:
            tagged_sents.append(nltk.pos_tag(word_tokenize(sent)))
        train_sents, test_sents = self.train_test_sentences(tagged_sents)
        train_tagger(tagging_model(train_sents, test_sents))

    def tag(sentences):
        with open('b_tagger.pkl', 'rb') as tag_in:
            tagger = pickle.load(tag_in)
            return tagger.tag(
                [w for sent in sentences for w in word_tokenize(sent)])


def train_test_sentences(tagged_sentences):
    size = int(len(tagged_sentences) * 0.9)
    train_sents = tagged_sentences[:size]
    test_sents = tagged_sentences[size:]
    return train_sents, test_sents


def tagging_model(train_sents, test_sents):
    train_token_list = []
    train_token_list.extend([w for sent in train_sents for w in sent])

    # take the most likely tag for freq words and create a model
    freq_words = nltk.FreqDist(
        [t[0] for t in train_token_list]).most_common()
    cfd_freq_words = nltk.ConditionalFreqDist(train_token_list)
    likely_tags = dict((word, cfd_freq_words[word].max())
                       for (word, _) in freq_words)
    return likely_tags


def train_tagger(tag_model):
    # default tagger
    default_tagger = nltk.DefaultTagger('NIL')

    # Unigram Tagger
    u_tagger = nltk.UnigramTagger(model=tag_model, backoff=default_tagger)

    # Bigram_tagger
    b_tagger = nltk.BigramTagger(model=tag_model, backoff=u_tagger)

    with open('b_tagger.pkl', 'wb') as out:
        pickle.dump(b_tagger, out, -1)
