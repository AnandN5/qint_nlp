from nltk.tag import brill
from nltk.tag.brill_trainer import BrillTaggerTrainer
import utils


class BrillTagger(object):

    def __init__(self):
        # bounds = [(1, end)]
        initial_tagger = get_initial_tagger()
        rules = brill.fntbl37()

        self.trainer = BrillTaggerTrainer(initial_tagger, rules,
                                          deterministic=True, trace=0)
        train_sents, test_sents = utils.training_testing_dataset()
        self.tagger = self.trainer.train(train_sents, max_rules=20)
        print('Brill tagger training completed')

    def tag(self, sent_tokens):
        tagged_sentences = []
        for sent in sent_tokens:
            tags = self.tagger.tag([w for w in sent])
            tagged_sentences.append(tags)
        return tagged_sentences


def get_initial_tagger():
    from TrainedTagger import CustomTrainedTagger
    return CustomTrainedTagger()
