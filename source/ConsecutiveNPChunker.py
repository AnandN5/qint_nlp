from nltk.chunk import ChunkParserI, conlltags2tree
from ChunkTagger import NgramChunkTagger


class ConsecutiveNPChunker(ChunkParserI):
    def __init__(self):
        self.tagger = NgramChunkTagger()

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        print(tagged_sents)
        return conlltags2tree(tagged_sents)
