# from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk


def process_data(file):
    with open(file, 'r') as f:
        sentences = sent_tokenize(f.read())
        tagged_sents = []
        trained_token_list = []
        for sent in sentences:
            tagged_sents.append(nltk.pos_tag(word_tokenize(sent)))
        train_sents, test_sents = train_test_sentences(tagged_sents)
        trained_token_list.extend([w for sent in train_sents for w in sent])

        # take the most likely tag for freq words and create a model
        freq_words = nltk.FreqDist(
            [t[0] for t in trained_token_list]).most_common()
        cfd_freq_words = nltk.ConditionalFreqDist(trained_token_list)
        likely_tags = dict((word, cfd_freq_words[word].max())
                           for (word, _) in freq_words)

        # default tagger
        default_tagger = nltk.DefaultTagger('NIL')

        # Unigram Tagger
        u_tagger = nltk.UnigramTagger(
            model=likely_tags, backoff=default_tagger)

        # Bigram_tagger
        b_tagger = nltk.BigramTagger(model=likely_tags, backoff=u_tagger)
        print('Evaluation: ', b_tagger.evaluate(test_sents))

        tagged_tests = b_tagger.tag(
            [w for sent in sentences for w in word_tokenize(sent)])
        print(tagged_tests[:50])


def train_test_sentences(tagged_sentences):
    size = int(len(tagged_sentences) * 0.9)
    train_sents = tagged_sentences[:size]
    test_sents = tagged_sentences[size:]
    return train_sents, test_sents


def main():
    filename = '/Users/qbuser/Documents/pythonWorks/BigDataWorks/qint_nlp/source/audit1.txt'
    process_data(file=filename)


if __name__ == "__main__":
    main()
