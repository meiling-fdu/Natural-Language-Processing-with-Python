import nltk
from nltk.corpus import brown, treebank, gutenberg
from nltk.tree import Tree
from collections import defaultdict
from timeit import timeit
import prettytable as pt

# 3
grammar1 = nltk.CFG.fromstring("""
  S -> NPV OR NAN | NON AND NPV
  NAN -> NPV AND NPV
  NON -> NPV OR NPV
  NPV -> NP V
  V -> "arrived" | "left" | "cheered"
  NP -> "Kim" | "Dana" | "everyone"
  OR -> "or"
  AND -> "and"
  """)
sent = "Kim arrived or Dana left and everyone cheered".split()
rd_parser = nltk.RecursiveDescentParser(grammar1)
for tree in rd_parser.parse(sent):
    print(tree)


# 4
help(Tree)


# 5
nan = Tree('NAN', [Tree('N', ['men']), Tree('AND', ['and']), Tree('N', ['women'])])
tree_1 = Tree('S', [Tree('ADJ', ['old']), nan])

tree_2 = Tree.fromstring("(S (ADJN (ADJ old) (N men)) (AND and) (N (N women)))")

print(tree_1)
print(tree_2)

tree_2.draw()

'The woman saw a man last Thursday'

tree_3 = Tree.fromstring("(S (NP (Det The)(N woman)) (VP (V saw) (NP (Det a)(N man)) (PP (ADJ last)(NOM Thursday))))")
tree_3.draw()


# 6

tree = Tree('S', [Tree('ADJN', [Tree('ADJ', ['old']), Tree('N', ['men'])]), Tree('AND', ['and']),
                  Tree('N', [Tree('N', ['women'])])])


def depth(tree):
    max_child_height = 0
    for child in tree:
        if isinstance(child, Tree):
            max_child_height = max(max_child_height, child.height())
        else:
            max_child_height = max(max_child_height, 1)
    return max_child_height


print(depth(tree))


# 15
grammar2 = nltk.CFG.fromstring("""
  S  -> NP VP
  NP -> Det Nom | PropN
  VP -> IV ADVS
  ADVS -> ADV ADV
  PropN -> 'Lee'
  IV ->  'ran'
  ADV -> 'away' | 'home'
  """)
sent = "Lee ran away home".split()
rd_parser = nltk.RecursiveDescentParser(grammar2)
for tree in rd_parser.parse(sent):
    print(tree)


# 16 (incomplete)
entries = nltk.corpus.ppattach.attachments('training')
table = defaultdict(lambda: defaultdict(set))
for entry in entries:
    key = entry.verb + '-' + entry.noun1
    table[key][entry.attachment].add(entry.noun1)

for key in sorted(table):
    if len(table[key]) > 1:
        print(key, 'N:', sorted(table[key]['N']), 'V:', sorted(table[key]['V']))


# 17
grammar2 = nltk.CFG.fromstring("""
  S  -> NP VP
  NP -> Det Nom | PropN
  VP -> IV ADVS
  ADVS -> ADV ADV
  PropN -> 'Lee'
  IV ->  'ran'
  ADV -> 'away' | 'home'
  """)
sent = "Lee ran away home".split()


def test_ct():
    ct_parser = nltk.ChartParser(grammar2)
    for tree in ct_parser.parse(sent):
        pass
        print(tree)
    return ct_parser


def test_rd():
    rd_parser = nltk.RecursiveDescentParser(grammar2)
    for tree in rd_parser.parse(sent):
        pass
        print(tree)
    return rd_parser


print(timeit(stmt=test_ct, number=1))
print(timeit(stmt=test_rd, number=1))


# 18
grammar = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP | IV ADVS
  ADVS -> ADV ADV
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> "John" | "Mary" | "Bob" | Det N | Det N PP | Det Nom | PropN
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "cat" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
  PropN -> 'Lee'
  IV ->  'ran'
  ADV -> 'away' | 'home'
  """)

sent1 = "Mary saw Bob".split()
sent2 = "Lee ran away home".split()
sent3 = "Mary saw a dog".split()


def test_rd1():
    rd_parser = nltk.RecursiveDescentParser(grammar)
    for tree in rd_parser.parse(sent1):
        pass
        # print(tree)
    return rd_parser


def test_rd2():
    rd_parser = nltk.RecursiveDescentParser(grammar)
    for tree in rd_parser.parse(sent2):
        pass
        # print(tree)
    return rd_parser


def test_rd3():
    rd_parser = nltk.RecursiveDescentParser(grammar)
    for tree in rd_parser.parse(sent3):
        pass
        # print(tree)
    return rd_parser


def test_ct1():
    ct_parser = nltk.ChartParser(grammar)
    for tree in ct_parser.parse(sent1):
        pass
        # print(tree)
    return ct_parser


def test_ct2():
    ct_parser = nltk.ChartParser(grammar)
    for tree in ct_parser.parse(sent2):
        pass
        # print(tree)
    return ct_parser


def test_ct3():
    ct_parser = nltk.ChartParser(grammar)
    for tree in ct_parser.parse(sent3):
        pass
        # print(tree)
    return ct_parser


def test_lc1():
    lc_parser = nltk.LeftCornerChartParser(grammar)
    for tree in lc_parser.parse(sent1):
        pass
        # print(tree)
    return lc_parser


def test_lc2():
    lc_parser = nltk.LeftCornerChartParser(grammar)
    for tree in lc_parser.parse(sent2):
        pass
        # print(tree)
    return lc_parser


def test_lc3():
    lc_parser = nltk.LeftCornerChartParser(grammar)
    for tree in lc_parser.parse(sent3):
        pass
        # print(tree)
    return lc_parser


rd1 = round(timeit(stmt=test_rd1, number=1), 6)
rd2 = round(timeit(stmt=test_rd2, number=1), 6)
rd3 = round(timeit(stmt=test_rd3, number=1), 6)
ct1 = round(timeit(stmt=test_ct1, number=1), 6)
ct2 = round(timeit(stmt=test_ct2, number=1), 6)
ct3 = round(timeit(stmt=test_ct3, number=1), 6)
lc1 = round(timeit(stmt=test_lc1, number=1), 6)
lc2 = round(timeit(stmt=test_lc2, number=1), 6)
lc3 = round(timeit(stmt=test_lc3, number=1), 6)


tb = pt.PrettyTable(["Parser", "Sent1", "Sent2", "Sent3", "Total"])
tb.add_row(['RecursiveDescentParser', rd1, rd2, rd3, round(rd1+rd2+rd3, 6)])
tb.add_row(['ChartParser', ct1, ct2, ct3, round(ct1+ct2+ct3, 6)])
tb.add_row(['LeftCornerChartParser', lc1, lc2, lc3, round(lc1+lc2+lc3, 6)])
tb.add_row(['Total', round(rd1+ct1+lc1, 6), round(rd2+ct2+lc2, 6), round(rd3+ct3+lc3, 6),
            round(rd1+ct1+lc1+rd2+ct2+lc2+rd3+ct3+lc3, 6)])
print(tb)


# 20
from nltk.draw.tree import draw_trees
nan = Tree('NAN', [Tree('N', ['men']), Tree('AND', ['and']), Tree('N', ['women'])])
tree_1 = Tree('S', [Tree('ADJ', ['old']), nan])
tree_2 = Tree.fromstring("(S (ADJN (ADJ old) (N men)) (AND and) (N (N women)))")
tree_3 = Tree.fromstring("(S (NP (Det The)(N woman)) (VP (V saw) (NP (Det a)(N man)) (PP (ADJ last)(NOM Thursday))))")
draw_trees(tree_1, tree_2, tree_3)


# 21
t = treebank.parsed_sents()[:100]
def filter(tree):
    child_nodes = [child.label() for child in tree
                   if isinstance(child, nltk.Tree) and child.height() <= 2]
    return (tree.label() == 'NP-SBJ')


subtrees = [subtree for tree in treebank.parsed_sents()
       for subtree in tree.subtrees(filter)]
for stree in subtrees:
    print(stree)


# 25
sents = gutenberg.sents('austen-emma.txt')
sorted_sents = sorted(sents, key=lambda x: len(x), reverse=True)
print(len(sorted_sents[0]), sorted_sents[0])
