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
        import pdb
        pdb.set_trace()
        for i, index in enumerate(rel_indexes):
            if type(index) == tuple:
                stop = index[0]
                offset = len(index)
                relation = ' '.join([sentence[i] for i in index])
            else:
                stop = index
                offset = 1
                relation = sentence[index]
            stop = index
            ent1 = ent2 if ent2 else sentence[1:stop]
            ent2 = sentence[stop + offset:rel_indexes[i + 1]
                            ] if i < len(rel_indexes) - 1 else sentence[stop + offset:]
            e1, e2, relation = utils.process_entities_list(ent1, ent2, relation)
            sent_entities.setdefault(relation, {'ent1': e1, 'ent2': e2})
        print(sent_entities)

    def relation_indexes(self, sentence):
        run = []
        result = [run]
        expect = 'RELATION'
        for i, subtree in enumerate(list(sentence.subtrees())):
            if (subtree.label() == expect):
                run.append(i)
            else:
                result.append(run) if len(run) > 0 else False
                run = []
        result = list(filter(None, result))
        result = [x[0] if len(x) < 2 else tuple(x) for x in result]
        return result
