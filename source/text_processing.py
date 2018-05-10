# from nltk.corpus import wordnet
from CombinationTagger import NGramTagger
from TrainedTagger import CustomTrainedTagger
from BrillTagger import BrillTagger
from ConsecutiveNPChunker import ConsecutiveNPChunker
from utils import tokenized_sents, noun_phrases
from RegexChunker import RegexChunker
from ChunkExtractor import ChunkExtractor
import os

# combo_tagger = NGramTagger()
trained_tagger = CustomTrainedTagger()
brill_tagger = BrillTagger()
taggers = ['combination', 'trained', 'brill']


def process_data(file):
    with open(file, 'r') as f:
        sentences = f.read()
        tagged_sents = tagged_text(sentences, taggers[1])
        print('tree format')
        extractor = ChunkExtractor()
        for i, t in enumerate(tagged_sents):
            print('{}.'.format(i))
            print(t)
            extractor.extract_entities(t)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iob_tagged_output.txt'), 'w') as out:
            for i, sent in enumerate(tagged_sents):
                out.write(str(i))
                out.write('\n')
                out.write(str(sent))
                out.write('\n')


def tagged_text(text, tagger):
    tagged_sentences = []
    tokenized = tokenized_sents(text)
    chunker = RegexChunker()
    if tagger == taggers[0]:
        # Combination tagger tagging
        # tagged_sentences = combo_tagger.tag(tokenized)
        print('combo tagger')

    elif tagger == taggers[1]:
        print('trained tagger')

        # Trained tagger tagging
        for sent in tokenized:
            tagged = trained_tagger.tag(sent)
            chunked = chunker.parse(tagged)
            tagged_sentences.append(chunked)
        return tagged_sentences

    elif tagger == taggers[2]:
        # Brill tagger
        tagged = brill_tagger.tag(tokenized)
        for sent in tagged:
            chunked = chunker.parse(sent)
            tagged_sentences.append(chunked)
        return tagged_sentences
        print('brill tagger')


def main():
    filename = '/Users/qbuser/Documents/pythonWorks/BigDataWorks/qint_nlp/source/data/audit_new.txt'
    process_data(file=filename)


if __name__ == "__main__":
    main()
