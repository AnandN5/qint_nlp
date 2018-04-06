from nltk.corpus import stopwords
# from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import re
import string


with open('/Users/qbuser/Documents/pythonWorks/BigDataWorks/nlp_work/qint_nlp/audit1.txt', 'r') as f:
    sentences = sent_tokenize(f.read())
    tokens_list = []
    sw = stopwords.words('english')
    for index, sent in enumerate(sentences):
        sent = re.sub('\W(?=\s|$)', '', sent)
        tokens = [w.lower() for w in word_tokenize(sent)]
        clean_tokens = tokens[:]

        # Removing stopwords
        for token in tokens:
            if token in sw or token in string.punctuation:
                clean_tokens.remove(token)
        tokens_list.append(nltk.pos_tag(clean_tokens))

    for index, item in enumerate(tokens_list):
        print('{}. \n'.format(index))
        grammar = r"""
                    # chunk determiner/possessive, adjectives and noun
                    audit_number: {<DT|PP\$>?<JJ>*<NN>}
                    non-confrmance: {<VBD>+}                # chunk sequences of proper nouns
                """

        cp = nltk.RegexpParser(grammar)
        print(type(cp.parse(item)))
