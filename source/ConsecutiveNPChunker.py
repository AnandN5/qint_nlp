from nltk.chunk import ChunkParserI, conlltags2tree
from ChunkTagger import NgramChunkTagger
# import nltk

grammar = r"""
    CLAUSE: {<NP><VP>}
"""


class ConsecutiveNPChunker(ChunkParserI):
    def __init__(self):
        self.tagger = NgramChunkTagger()

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        tagged_sents = conlltags2tree(tagged_sents)

        # Nested chunked tags for CLAUSING

        # cp = nltk.RegexpParser(grammar)
        # tagged_sents = cp.parse(tagged_sents)
        return tagged_sents
