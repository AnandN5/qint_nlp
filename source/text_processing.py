# from nltk.corpus import wordnet
from CombinationTagger import NGramTagger
from TrainedTagger import CustomTrainedTagger
from BrillTagger import BrillTagger
from ConsecutiveNPChunker import ConsecutiveNPChunker
from utils import tokenized_sents
import os

# combo_tagger = NGramTagger()
trained_tagger = CustomTrainedTagger()
# brill_tagger = BrillTagger()
taggers = ['combination', 'trained', 'brill']


def process_data(file):
    with open(file, 'r') as f:
        sentences = f.read()
        # tagged_sents = combo_tagger.tag(sentences)
        # tagged_sents = trained_tagger.tag(sentences)
        # tagged_sents = brill_tagger.tag(sentences)
        tagged_sents = tagged_text(sentences, taggers[1])
        print('tree format')
        for i, t in enumerate(tagged_sents):
            print('{}.'.format(i))
            print(t)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iob_tagged_output.txt'), 'w') as out:
            for i, sent in enumerate(tagged_sents):
                out.write(str(i) + '\n')
                out.write(str(sent))


def tagged_text(text, tagger):
    tagged_sentences = []
    tokenized = tokenized_sents(text)
    if tagger == taggers[0]:
        # Combination tagger tagging
        # tagged_sentences = combo_tagger.tag(tokenized)
        print('combo tagger')

    elif tagger == taggers[1]:
        print('trained tagger')
        # Trained tagger tagging
        chunker = ConsecutiveNPChunker()
        for sent in tokenized:
            tagged = trained_tagger.tag(sent)
            chunked = chunker.parse(tagged)
            tagged_sentences.append(chunked)
        return tagged_sentences
        print('trained tagger')

    elif tagger == taggers[2]:
        # Brill tagger
        # tagged_sentences = brill_tagger.tag(tokenized)
        # return tagged_sentences
        print('brill tagger')


def main():
    filename = '/Users/qbuser/Documents/pythonWorks/BigDataWorks/qint_nlp/source/data/audit_new.txt'
    process_data(file=filename)


if __name__ == "__main__":
    main()
