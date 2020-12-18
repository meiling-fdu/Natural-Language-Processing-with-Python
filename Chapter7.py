import nltk
from nltk.corpus import conll2000, conll2002
import re
#
#
# # Information Extraction
# locs = [('Omnicom', 'IN', 'New York'),
#         ('DDB Needham', 'IN', 'New York'),
#         ('Kaplan Thaler Group', 'IN', 'New York'),
#         ('BBDO South', 'IN', 'Atlanta'),
#         ('Georgia-Pacific', 'IN', 'Atlanta')]
# query = [e1 for (e1, rel, e2) in locs if e2 == 'Atlanta']
# print(query)
#
#
# def ie_preprocess(document):
#     sentences = nltk.sent_tokenize(document)[1]
#     sentences = [nltk.word_tokenize(sent) for sent in sentences][2]
#     sentences = [nltk.pos_tag(sent) for sent in sentences][3]
#
#
# # Chunking
# sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
#             ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
# grammar = "NP: {<DT>?<JJ>*<NN>}"
# cp = nltk.RegexpParser(grammar)
# result = cp.parse(sentence)
# print(result)
# result.draw()
#
#
# nltk.app.chunkparser()
# s1 = [('his', "PRP$"), ('Mansion', "NNP"), ('House', "NNP"), ('speech', "NN")]
# s2 = [('the', "DT"), ('price', "NN"), ('cutting', "VBG")]
# s3 = [('3', "CD"), ('%', "NN"), ('to', "TO"), ('4', "CD"), ('%', "NN")]
# s4 = [('more', "JJR"), ('than', "IN"), ('4', "CD"), ('%', "NN")]
# s5 = [('the', "DT"), ('fastest', "JJS"), ('developing', "VBG"), ('trends', "NNS")]
# s6 = [("'s", "POS"), ('skill', "NN")]
# sents = [s1, s2, s3, s4, s5, s6]
#
# grammar = r"""
# NP: {<CD><NN>}
#     {<DT|PRP\$|POS>?<JJ.*|JJS>*<VBG>*<NN.*>+<VBG>*}
#
# """
# cp = nltk.RegexpParser(grammar)
# result = cp.parse_sents(sents)
# for rst in result:
#     print(rst)
#     rst.draw()
#
#
# grammar = r"""
# NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
#     {<NNP>+}                # chunk sequences of proper nouns
# """
# cp = nltk.RegexpParser(grammar)
# sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"),
#             ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
# print(cp.parse(sentence))
#
#
# nouns = [("money", "NN"), ("market", "NN"), ("fund", "NN")]
# grammar = "NP: {<NN><NN>}  # Chunk two consecutive nouns"
# cp = nltk.RegexpParser(grammar)
# print(cp.parse(nouns, trace=1))
#
# grammar = "NP: {<NN>+}  # Chunk consecutive nouns"
# cp = nltk.RegexpParser(grammar)
# print(cp.parse(nouns, trace=1))
#
#
# cp = nltk.RegexpParser('CHUNK: {<V.*> <TO> <V.*>}')
# brown = nltk.corpus.brown
# for sent in brown.tagged_sents():
#     tree = cp.parse(sent)
#     for subtree in tree.subtrees():
#         if subtree.label() == 'CHUNK':
#             print(subtree)
#
#
# def find_chunks(chunk):
#     brown = nltk.corpus.brown
#     cp = nltk.RegexpParser(chunk)
#     for sent in brown.tagged_sents():
#         tree = cp.parse(sent)
#         for subtree in tree.subtrees():
#             if subtree.label() == 'NUMBER':
#                 print(subtree)
# # find_chunks('CHUNK: {<V.*> <TO> <V.*>}')
# # find_chunks('NOUNS: {<N.*>{4,}}')
# # find_chunks('PONS: {<DT|PP\$><JJ>+<NN>}')
# # find_chunks('NUMBER: {<CD.*>+}')
#
#
# grammar = r"""
#   NP:
#     {<.*>+}          # Chunk everything
#     }<VBD|IN>+{      # Chink sequences of VBD and IN
#   """
# sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
#        ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
# cp = nltk.RegexpParser(grammar)
# print(cp.parse(sentence))
#
#
# # Developing and Evaluating Chunkers
# text = '''
# he PRP B-NP
# accepted VBD B-VP
# the DT B-NP
# position NN I-NP
# of IN B-PP
# vice NN B-NP
# chairman NN I-NP
# of IN B-PP
# Carlyle NNP B-NP
# Group NNP I-NP
# , , O
# a DT B-NP
# merchant NN I-NP
# banking NN I-NP
# concern NN I-NP
# . . O
# '''
# nltk.chunk.conllstr2tree(text, chunk_types=['NP']).draw()
#
#
# print(conll2000.chunked_sents('train.txt')[99])
# print(conll2000.chunked_sents('train.txt', chunk_types=['NP'])[99])
#
#
# cp = nltk.RegexpParser("")
# test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
# print(cp.evaluate(test_sents))
#
# grammar = r"NP: {<[CDJNP].*>+}"
# cp = nltk.RegexpParser(grammar)
# print(cp.evaluate(test_sents))
#
#
# class UnigramChunker(nltk.ChunkParserI):
#     def __init__(self, train_sents):
#         train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
#                       for sent in train_sents]
#         self.tagger = nltk.UnigramTagger(train_data)
#
#     def parse(self, sentence):
#         pos_tags = [pos for (word,pos) in sentence]
#         tagged_pos_tags = self.tagger.tag(pos_tags)
#         chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
#         conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
#                      in zip(sentence, chunktags)]
#         return nltk.chunk.conlltags2tree(conlltags)
#
# test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
# train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
# unigram_chunker = UnigramChunker(train_sents)
# print(unigram_chunker.evaluate(test_sents))
#
# postags = sorted(set(pos for sent in train_sents
#                      for (word, pos) in sent.leaves()))
# print(unigram_chunker.tagger.tag(postags))
#
#
# class BigramChunker(nltk.ChunkParserI):
#     def __init__(self, train_sents):
#         train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
#                       for sent in train_sents]
#         self.tagger = nltk.BigramTagger(train_data)
#
#     def parse(self, sentence):
#         pos_tags = [pos for (word,pos) in sentence]
#         tagged_pos_tags = self.tagger.tag(pos_tags)
#         chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
#         conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
#                      in zip(sentence, chunktags)]
#         return nltk.chunk.conlltags2tree(conlltags)
#
# bigram_chunker = BigramChunker(train_sents)
# print(bigram_chunker.evaluate(test_sents))
#
#
import os

os.environ["MEGAM"] = 'D:\Anaconda3_32_bit\Lib\site-packages\MEGAM\megam-64'


def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    return {"pos": pos}


class ConsecutiveNPChunkTagger(nltk.TaggerI):

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        # self.classifier = nltk.NaiveBayesClassifier.train(train_set)
        self.classifier = nltk.MaxentClassifier.train(
            train_set, algorithm='MEGAM', trace=0)  #

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c) for (w, t, c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))
#
#
# def npchunk_features(sentence, i, history):
#     word, pos = sentence[i]
#     if i == 0:
#         prevword, prevpos = "<START>", "<START>"
#     else:
#         prevword, prevpos = sentence[i-1]
#     return {"pos": pos, "prevpos": prevpos}
#
# def npchunk_features(sentence, i, history):
#     word, pos = sentence[i]
#     if i == 0:
#         prevword, prevpos = "<START>", "<START>"
#     else:
#         prevword, prevpos = sentence[i-1]
#     return {"pos": pos, "word": word, "prevpos": prevpos}
#
# def npchunk_features(sentence, i, history):
#     word, pos = sentence[i]
#     if i == 0:
#         prevword, prevpos = "<START>", "<START>"
#     else:
#         prevword, prevpos = sentence[i-1]
#     if i == len(sentence)-1:
#         nextword, nextpos = "<END>", "<END>"
#     else:
#         nextword, nextpos = sentence[i+1]
#     return {"pos": pos,
#             "word": word,
#             "prevpos": prevpos,
#             "nextpos": nextpos,
#             "prevpos+pos": "%s+%s" % (prevpos, pos),
#             "pos+nextpos": "%s+%s" % (pos, nextpos),
#             "tags-since-dt": tags_since_dt(sentence, i)
#             }
# def tags_since_dt(sentence, i):
#     tags = set()
#     for word, pos in sentence[:i]:
#         if pos == 'DT':
#             tags = set()
#         else:
#             tags.add(pos)
#     return '+'.join(sorted(tags))
#
#
# # Building Nested Structure with Cascaded Chunkers
# grammar = r"""
#   NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
#   PP: {<IN><NP>}               # Chunk prepositions followed by NP
#   VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
#   CLAUSE: {<NP><VP>}           # Chunk NP, VP
#   """
# cp = nltk.RegexpParser(grammar)
# sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
#             ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]
# print(cp.parse(sentence))
#
# sentence = [("John", "NNP"), ("thinks", "VBZ"), ("Mary", "NN"),
#             ("saw", "VBD"), ("the", "DT"), ("cat", "NN"), ("sit", "VB"),
#             ("on", "IN"), ("the", "DT"), ("mat", "NN")]
# print(cp.parse(sentence))
# cp = nltk.RegexpParser(grammar, loop=3)
# print(cp.parse(sentence))
#
#
# # Tree
# tree1 = nltk.Tree('NP', ['Alice'])
# print(tree1)
# tree2 = nltk.Tree('NP', ['the', 'rabbit'])
# print(tree2)
# tree3 = nltk.Tree('VP', ['chased', tree2])
# tree4 = nltk.Tree('S', [tree1, tree3])
# print(tree4)
# print(tree4[1])
# print(tree4[1].label())
# print(tree4.leaves())
# print(tree4[1][1].label())
# print(tree4[1][1][1])
# tree4.draw()
#
#
# def traverse(t):
#     try:
#         t.label()
#     except AttributeError:
#         print(t, end=" ")
#     else:
#         # Now we know that t.node is defined
#         print('(', t.label(), end=" ")
#         for child in t:
#             traverse(child)
#         print(')', end=" ")
#
# t = nltk.Tree.fromstring('(S (NP Alice) (VP chased (NP the rabbit)))')
# traverse(t)
#
#
# # Named Entity Recognition
# sent = nltk.corpus.treebank.tagged_sents()[0]
# print(nltk.ne_chunk(sent, binary=True))
# print(nltk.ne_chunk(sent, binary=False))
#
#
# # Relation Extraction
# IN = re.compile(r'.*\bin\b(?!\b.+ing)')
# for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
#     for rel in nltk.sem.extract_rels('ORG', 'LOC', doc,
#                                      corpus='ieer', pattern = IN):
#         print(nltk.sem.rtuple(rel))
#
#
# vnv = """
# (
# is/V|    # 3rd sing present and
# was/V|   # past forms of the verb zijn ('be')
# werd/V|  # and also present
# wordt/V  # past of worden ('become)
# )
# .*       # followed by anything
# van/Prep # followed by van ('of')
# """
# VAN = re.compile(vnv, re.VERBOSE)
# for doc in conll2002.chunked_sents('ned.train'):
#     for rel in nltk.sem.extract_rels('PER', 'ORG', doc,
#                                      corpus='conll2002', pattern=VAN):
#         # print(nltk.sem.clause(rel, relsym="VAN"))
#         print(nltk.rtuple(rel, lcon=True, rcon=True))
#
#
#
#
#
#
#
#
#
#
#
#
