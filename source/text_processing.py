# from nltk.corpus import wordnet
from CombinationTagger import NGramTagger

tagger = NGramTagger()


def process_data(file):
    with open(file, 'r') as f:
        sentences = f.read()
        tagged_sents = tagger.tag(sentences)
        print(tagged_sents[:500])


def main():
    filename = '/Users/qbuser/Documents/pythonWorks/BigDataWorks/qint_nlp/source/data/audit_new.txt'
    process_data(file=filename)


if __name__ == "__main__":
    main()
