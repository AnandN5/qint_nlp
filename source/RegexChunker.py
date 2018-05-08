from nltk import RegexpParser

# grammar = """ENTITY : {<DT>?<JJ>*<NN.*>}
#                       {<DT>?<JJ>*<NN.*>+<CD>?}
#                       {<NN.*>}
#              RELATION: {<V.*>+}
#                        {<V.*>+<IN>*}
#                        {<DT>?<JJ>*<NN.*>+}
# """
grammar = """ENTITY : {<DT>?<JJ>*<NN.*>+<CD>*}
     {<DT>*<JJ>*<CD>*}
     {<NN.*>}
     RELATION: {<V.*>+<IN>*}
     {<DT>?<JJ>*<NN.*>+}
     """


class RegexChunker:
    def __init__(self):
        self.grammar = grammar
        self.parser = RegexpParser(grammar)

    def parse(self, tagged_sent):
        return self.parser.parse(tagged_sent)
