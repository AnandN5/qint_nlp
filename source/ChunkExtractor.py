import utils


class ChunkExtractor(object):
    def __init__(self):
        self.entities_dict = {}

    def extract_entities(self, sentence):
        sent_entities = {}
        ent1 = []
        ent2 = []
        rel_indexes = self.relation_indexes(sentence)
        sentence = [sent.leaves() for sent in sentence.subtrees()]
        if rel_indexes:
            for i, index in enumerate(rel_indexes):
                stop = index
                ent1 = ent2 if ent2 else sentence[0:stop]
                relation = sentence[stop]
                ent2 = sentence[stop + 1:rel_indexes[i + 1]
                                ] if i < len(rel_indexes) - 1 else sentence[stop + 1:]
                ent1, ent2 = utils.process_entities_list(ent1, ent2)
                sent_entities.setdefault(relation, {'ent1': ent1, 'ent2': ent2})
        else:
            pass
        print(sent_entities)

    def relation_indexes(self, sentence):
        rel_indexes = []
        # import pdb
        # pdb.set_trace()
        for i, subtree in enumerate(list(sentence.subtrees())):
            if subtree.label() == 'RELATION':
                rel_indexes.append(i)
        print(rel_indexes)
