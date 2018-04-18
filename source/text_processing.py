# from nltk.corpus import wordnet
from CombinationTagger import NGramTagger
from TrainedTagger import CustomTrainedTagger

combo_tagger = NGramTagger()
trained_tagger = CustomTrainedTagger()


def process_data(file):
    with open(file, 'r') as f:
        sentences = f.read()
        tagged_sents = combo_tagger.tag(sentences)
        # tagged_sents = trained_tagger.tag(sentences)
        print(tagged_sents)


def main():
    filename = '/Users/qbuser/Documents/pythonWorks/BigDataWorks/qint_nlp/source/data/audit_new.txt'
    process_data(file=filename)


if __name__ == "__main__":
    main()
