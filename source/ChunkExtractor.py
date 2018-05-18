from key_recognizer import KeyRecognizer
import utils


class ChunkExtractor(object):
    def __init__(self):
        self.entities_dict = {}
        self.recognizer = KeyRecognizer()
        self.recognizer.train()

    def extract_entities(self, sentence):
        sent_entities = {}
        ent1 = []
        ent2 = []
        rel_indexes = self.__relation_indexes(sentence)
        print(rel_indexes)
        sentence = [sent.leaves() for sent in sentence.subtrees()]
        # sentence = sentence[1:]
        for i, index in enumerate(rel_indexes):

            stop, offset, relation, next_index = self.__sent_traverse_det(
                i, index, sentence, rel_indexes)
            ent1 = ent2 if ent2 else sentence[1:stop]
            ent2 = sentence[stop + offset:next_index]
            e1, e2, relation = utils.process_entities_list(
                ent1, ent2, relation)
            sent_entities.setdefault(relation, {'ent1': e1, 'ent2': e2})
        return sent_entities

    # def extract_all_entities(self, tagged_text):
    #     entities = []
    #     for sent in tagged_text:
    #         for subtree in list(sent.subtrees()):
    #             if subtree.label() == 'ENTITY':
    #                 for leaf in subtree.leaves():
    #                     # entities.append(
    #                     #     subtree.leaves()) if subtree.label() == 'ENTITY' else ''
    #                     entities.append(
    #                         leaf) if leaf[1] == 'JJ' else ''
    #     utils.write_to_file(entities)

        # return entities

    def extract_all_entities(self, tagged_text):
        entities = []
        audits = []
        non_confs = []
        for sent in tagged_text:
            a, c = self.recognizer.tag(sent)
            audits.append(a)
            non_confs.append(c)
        print(self.recognizer.classifier.show_most_informative_features())
        for a in list(filter(None, audits)):
            print(a)


    def __relation_indexes(self, sentence):
        index = []
        result = [index]
        expect = 'RELATION'
        for i, subtree in enumerate(list(sentence.subtrees())):
            if (subtree.label() == expect):
                index.append(i)
            else:
                result.append(index) if len(index) > 0 else False
                index = []
        result = list(filter(None, result))
        result = [x[0] if len(x) < 2 else tuple(x) for x in result]
        return result

    def __sent_traverse_det(self, i, index, sent, rel_indexes):
        if type(index) == tuple:
            rel = []
            stop = index[0]
            offset = len(index)
            for j in index:
                rel.extend(sent[j])
            relation = rel
        else:
            stop = index
            offset = 1
            relation = sent[index]

        def get_next_index():
            if i < len(rel_indexes) - 1:
                if type(rel_indexes[i + 1]) == tuple:
                    return rel_indexes[i + 1][0]
                else:
                    return rel_indexes[i + 1]
            else:
                return None

        next_index = get_next_index()
        return stop, offset, relation, next_index
