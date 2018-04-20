from nltk.tag import brill
import utils


def initial_tagger():
    import pickle
    with open('trained_tagger.pkl', 'rb') as inp:
        tagger = pickle.load(inp)
        return tagger


class BrillTagger:

    def __init__(self):
        self.initial_tagger = initial_tagger()
        self.rules = [
            brill.SymmetricProximateTokensTemplate(
                brill.ProximateTagsRule, (1, 1)),
            brill.SymmetricProximateTokensTemplate(
                brill.ProximateTagsRule, (2, 2)),
            brill.SymmetricProximateTokensTemplate(
                brill.ProximateTagsRule, (1, 2)),
            brill.SymmetricProximateTokensTemplate(
                brill.ProximateTagsRule, (1, 3)),
            brill.SymmetricProximateTokensTemplate(
                brill.ProximateWordsRule, (1, 1)),
            brill.SymmetricProximateTokensTemplate(
                brill.ProximateWordsRule, (2, 2)),
            brill.SymmetricProximateTokensTemplate(
                brill.ProximateWordsRule, (1, 2)),
            brill.SymmetricProximateTokensTemplate(
                brill.ProximateWordsRule, (1, 3)),
            brill.ProximateTokensTemplate(
                brill.ProximateTagsRule, (-1, -1), (1, 1)),
            brill.ProximateTokensTemplate(
                brill.ProximateWordsRule, (-1, -1), (1, 1))
        ]

        self.trainer = brill.FastBrillTaggerTrainer(
            self.initial_tagger, self.rules)
        train_sents, test_sents = utils.training_testing_dataset()
        self.tagger = self.trainer.train(train_sents, max_rules=100, min_score=3)
