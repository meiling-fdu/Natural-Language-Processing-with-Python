import nltk
from nltk import chunk, tree
from nltk.corpus import conll2000, brown, treebank_chunk, treebank
from nltk.chunk import ChunkScore

# 1 ref: https://linguistics.stackexchange.com/questions/28482/why-are-three-tags-necessary-for-the-iob-format-what-problem-would-be-caused-if
# IO format can’t represent two entities next to each other, because there’s no boundary tag.


# 2
grammar = r"""
PNP: {<DT|PP\$|CD>?<JJ>*<NNS>}   # chunk determiner/possessive, adjectives and plural noun
    {<NNP>+}                # chunk sequences of proper nouns
"""
cp = nltk.RegexpParser(grammar)
s1 = [("two", "CD"), ("weeks", "NNS")]
s2 = [("many", "JJ"), ("researchers", "NNS")]
s3 = [("both", "DT"), ("new", "JJ"), ("positions", "NNS")]
sents = [s1, s2, s3]
result = cp.parse_sents(sents)
for rst in result:
    print(rst)
    rst.draw()


# 3
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
grammar = r"NP: {<[CDJNP].*>+}"
cp = nltk.RegexpParser(grammar, loop=1)
print(cp.evaluate(test_sents))


# 4
grammar = r"""
  NP:
    {<.*>+}          # Chunk everything
    }<VBD|IN>+{      # Chink sequences of VBD and IN
  """
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
       ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
cp = nltk.RegexpParser(grammar)
print(cp.parse(sentence))


# 5
grammar = r"""
VNGNP: {<DT|PP\$|CD>?<JJ>*<NN|VBG>+}
"""
cp = nltk.RegexpParser(grammar)
s1 = [("the", "DT"), ("receiving", "VBG"), ("end", "NN")]
s2 = [("assistant", "NN"), ("managing", "VBG"), ("editor", "NN")]
s3 = [("their", "PP$"), ("new", "JJ"), ("swimming", "VBG"), ("pool", "NN")]
sents = [s1, s2, s3]
result = cp.parse_sents(sents)
for rst in result:
    print(rst)
    rst.draw()


# 6
grammar = r"""
CNP: {<DT>?<PRP\$>?<JJ>*<NNP|NN|NNS>+<CC><NNP|NN|NNS>+}
"""
cp = nltk.RegexpParser(grammar)
s1 = [("July", "NNP"), ("and", "CC"), ("August", "NNP")]
s2 = [("all", "DT"), ("your", "PRP$"), ("managers", "NNS"), ("and", "CC"), ("supervisors", "NNS")]
s3 = [("company", "NN"), ("courts", "NNS"), ("and", "CC"), ("adjudicators", "NNS")]
sents = [s1, s2, s3]
result = cp.parse_sents(sents)
for rst in result:
    print(rst)
rst.draw()


# 7 ref: https://www.nltk.org/api/nltk.chunk.html
class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])[:100]
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])[:100]
unigram_chunker = UnigramChunker(train_sents)
print("Unigram_chunker: ", unigram_chunker.evaluate(test_sents))

grammar = r"""
NP: {<DT>?<PRP\$>?<JJ>*<NNP|NN|NNS>+}
"""
cp = nltk.RegexpParser(grammar)
chunkscore = ChunkScore()
for correct in test_sents:
    guess = cp.parse(correct.leaves())
    chunkscore.score(correct, guess)
print('The misssed:', chunkscore.missed())
print('The incorrect:', chunkscore.incorrect())
print(chunkscore.accuracy())


# 8
RegexpChunk = r"""
NP: {<DT>?<PRP\$>?<CD>?<JJ>*<NNP|NN|NNS>+}
    {<[CDJNP].*>+}
    {<DT|PRP\$|POS>?<JJ.*|JJS>*<VBG>*<NN.*>+<VBG>*}
    }<VBD|IN>+{
"""


test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
cp = nltk.RegexpParser(RegexpChunk)
print(cp.evaluate(test_sents))

chunkscore = ChunkScore()
for correct in test_sents:
    guess = cp.parse(correct.leaves())
    chunkscore.score(correct, guess)
print(chunkscore.accuracy())


# 10
test_sents = conll2000.chunked_sents('test.txt')  #, chunk_types=['NP']
train_sents = conll2000.chunked_sents('train.txt')


class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


class TrigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.TrigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

bigram_chunker = BigramChunker(train_sents)
print(bigram_chunker.evaluate(test_sents))
trigram_chunker = TrigramChunker(train_sents)
print(trigram_chunker.evaluate(test_sents))


# 11
class BrillChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_sents = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        trainer = nltk.brill_trainer.BrillTaggerTrainer(
            initial_tagger=nltk.DefaultTagger('O'),
            templates=nltk.tag.brill.nltkdemo18(),
            trace=3,
            deterministic=True
        )
        self.tagger = trainer.train(train_sents, max_rules=20, min_score=2, min_acc=None)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
brill_chunker = BrillChunker(train_sents)
print("Brill_chunker: ", brill_chunker.evaluate(test_sents))

RegexpChunk = r"""
NP: {<DT>?<PRP\$>?<CD>?<JJ>*<NNP|NN|NNS>+}
    {<[CDJNP].*>+}
    {<DT|PRP\$|POS>?<JJ.*|JJS>*<VBG>*<NN.*>+<VBG>*}
    }<VBD|IN>+{
"""
cp = nltk.RegexpParser(RegexpChunk)
print(cp.evaluate(test_sents))


# 12
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
# print(test_sents[0], test_sents[0].leaves())

test_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
             for sent in test_sents]
print(test_data[0])

cfd = nltk.ConditionalFreqDist(
    ((x[1], y[1], z[0]), z[1])
    for sent in test_data
    for x, y, z in nltk.trigrams(sent))
ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
print(sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N())


# 13 (incomplete)
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])

tag_seqs = [[((w, t), c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
            for sent in test_sents]


# 14 (incomplete)
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])

print(train_sents[0])

for sent in train_sents[:1]:
    for subtree in sent.subtrees():
        if subtree.label() == 'NP':
            print(subtree)


# 15
grammar = r"""
NP: {<DT|PP\$>?<JJ>*<NN.*>+}
VP: {<VB.*><RP|IN>}
CLAUSE: {<VP><NP>}  # Chunk NP, VP
"""
cp = nltk.RegexpParser(grammar)
# sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"),
#             ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
            ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]
sent = cp.parse(sentence)
print(sent)

for subtree in sent.subtrees():
    if subtree.label() == 'CLAUSE':
        print(subtree)
        tup_vb = subtree[0][0][0]
        tup_pp = subtree[0][1][0]
        tup_np = subtree[1].label()
        tup = (tup_vb, tup_pp, tup_np)
        print(tup)


# 16 (incomplete)
print(treebank_chunk.fileids())
sents = treebank_chunk.chunked_sents('wsj_0199.pos')


def chunk2brackets(tree):
    tree.pprint()


def chunk2iob(tree):
    string = nltk.chunk.tree2conllstr(tree)
    print(string)


chunk2brackets(sents[0])
chunk2iob(sents[0])