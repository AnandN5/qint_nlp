class ChunkExtractor(object):
    def __init__(self):
        self.entities_dict = {}

    def extract_entities(self, sentence):
        sent_entities = {}
        for subtree in sentence.subtrees
