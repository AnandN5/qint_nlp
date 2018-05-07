import os
from nltk import word_tokenize, sent_tokenize, pos_tag, corpus
from nltk.chunk import tree2conlltags

train_data_dir = '/Users/qbuser/Documents/pythonWorks/BigDataWorks/qint_nlp/source/data/training_resource'


def is_txt_file(filename):
    return filename.endswith('.txt')


def training_testing_dataset():
    tagged_sents = []
    try:
        files = [f for f in os.listdir(train_data_dir) if os.path.isfile(
            os.path.join(train_data_dir, f))]
    except Exception as e:
        raise e

    for file in files:
        file_path = os.path.join(train_data_dir, file)
        if is_txt_file(file_path):
            with open(file_path, 'r') as f:
                sents = sent_tokenize(f.read())
                for sent in sents:
                    tagged_sents.append(pos_tag(word_tokenize(sent)))

    size = int(len(tagged_sents) * 0.9)
    train_sents = tagged_sents[:size]
    test_sents = tagged_sents[size:]
    return train_sents, test_sents


def tokenized_sents(text):
    sentences = text.split('\n')
    tokenized = [word_tokenize(sent) for sent in sentences]
    tokenized = list(filter(None, tokenized))
    return tokenized


def chunked_training_dataset():
    conll_train = corpus.conll2000.chunked_sents('train.txt')
    conll_test = corpus.conll2000.chunked_sents('test.txt')
    tag_sents_train = [tree2conlltags(tree) for tree in conll_train]
    train_tags = [[(t, c) for (w, t, c) in chunked_sents]
                  for chunked_sents in tag_sents_train]

    tag_sents_test = [tree2conlltags(tree) for tree in conll_test]
    test_tags = [[(t, c) for (w, t, c) in chunked_sents]
                 for chunked_sents in tag_sents_test]
    return train_tags, test_tags


def tags_since_dt(sent_tags, i):
    tags = set()
    for tag in sent_tags[:i]:
        if tag == 'DT':
            tags = set()
        else:
            tags.add(tag)
    return '+'.join(sorted(tags))


def noun_phrases(sent_tree):
    NPs = []
    for subtree in sent_tree.subtrees(lambda t: t.label() == 'NP'):
        NPs.append(subtree.leaves())
    return NPs


def verb_phrases(sent_trees):
    VPs = []
    for subtree in sent_trees.subtrees(lambda t: t.node == 'VP'):
        VPs.append(subtree.leaves())
    return VPs
