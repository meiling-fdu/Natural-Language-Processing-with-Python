import copy
import nltk
import pylab
from numpy import linalg
import numpy as np
from inspect import trace
from babel._compat import izip
from nltk import ngrams
import re
import networkx as nx
import matplotlib
from nltk.corpus import wordnet as wn

foo = ['Monty', 'Python']
bar = foo
foo[1] = 'Bodkin'
print(bar)

empty = []
nested = [empty, empty, empty]
print(nested)
nested[1].append('Python')
print(nested)

nested = [[]] * 3
print(nested)
nested[1] = 'lml'
print(nested)
print(id(nested[0]), id(nested[1]), id(nested[2]))

foo = ['Monty', 'Python']
bar = foo[:]
print(bar)
foo[1] = 'lml'
print(bar)
bar = copy.deepcopy(foo)
print(bar)
foo[1] = 'good'
print(bar)

size = 5
python = ['Python']
snake_nest = [python] * size
print(snake_nest)
print(snake_nest[0] == snake_nest[1] == snake_nest[2] == snake_nest[3] == snake_nest[4])
print(snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4])
import random

position = random.choice(range(size))
snake_nest[position] = ['Python']
print(snake_nest)
print(snake_nest[0] == snake_nest[1] == snake_nest[2] == snake_nest[3] == snake_nest[4])
print(snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4])
print([id(snake) for snake in snake_nest])

mixed = ['cat', '', ['dog'], []]
for element in mixed:
    if element:
        print(element)

animals = ['cat', 'dog']
if 'cat' in animals:
    print(1)
elif 'dog' in animals:
    print(2)

sent = ['No', 'good', 'fish', 'goes', 'anywhere', 'without', 'a', 'porpoise', '.']
print(all(len(w) > 4 for w in sent))
print(any(len(w) > 4 for w in sent))

t = 'walk', 'fem', 3
print(t)
print(t[0])
print(t[1:])
print(len(t))

t = 'snark',
et = ()
print('tuple: ', t, 'empty tuple:', et)

raw = 'I turned off the spectroroute'
text = ['I', 'turned', 'off', 'the', 'spectroroute']
pair = (6, 'turned')
print(raw[2], text[3], pair[1])
print(raw[-3:], text[-3:], pair[-3:])
print(len(raw), len(text), len(pair))

text_set = set('what a wonderful day!')
for i in text_set:
    print(i)
print(reversed(sorted(set(text_set))))

raw = 'Red lorry, yellow lorry, red lorry, yellow lorry.'
text = nltk.word_tokenize(raw)
fdist = nltk.FreqDist(text)
print(list(fdist))
for key in fdist:
    print(fdist[key], end=', ')

words = ['I', 'turned', 'off', 'the', 'spectroroute']
words[2], words[3], words[4] = words[3], words[4], words[2]
print(words)

words = ['I', 'turned', 'off', 'the', 'spectroroute']
tags = ['noun', 'verb', 'prep', 'det', 'noun']
zips = zip(words, tags)
print(zips, )
lists = list(enumerate(words))
print(lists)

text = nltk.corpus.nps_chat.words()
cut = int(0.9 * len(text))
training_data, test_data = text[:cut], text[cut:]
print(text == training_data + test_data)
print(len(training_data) / len(test_data))

words = 'I turned off the spectroroute'.split()
wordlens = [(len(word), word) for word in words]
wordlens.sort()
string = ' '.join(w for (_, w) in wordlens)
print(string)

lexicon = [
    ('the', 'det', ['Di:', 'D@']),
    ('off', 'prep', ['Qf', 'O:f'])
]
lexicon.sort()
print(lexicon)
lexicon[1] = ('turned', 'VBD', ['t3:nd', 't3`nd'])
print(lexicon)
del lexicon[0]
print(lexicon)
lexicon_tuple = tuple(lexicon)
lexicon_tuple.sort()
lexicon_tuple[1] = ('off', 'prep', ['Qf', 'O:f'])
del lexicon_tuple[0]

text = '''
"When I use a word," Humpty Dumpty said in rather a scornful tone,
"it means just what I choose it to mean - neither more nor less."
'''
words = [w.lower() for w in nltk.word_tokenize(text)]
print(words)
print(max([w.lower() for w in nltk.word_tokenize(text)]))
# generator expression
print(max(w.lower() for w in nltk.word_tokenize(text)))

sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
print(ngrams(sent, 3))


def get_text(file):
    """Read text from a file, normalizing whitespace and stripping HTML markup."""
    text = open(file, encoding='utf8').read()
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    return text


print(get_text('3_37_file.html'))
print(help(get_text))


def repeat(msg, num):
    return ' '.join([msg] * num)


monty = 'Monty Python'
print(repeat(monty, 3))


def set_up(word, properties):
    word = 'lolcat'
    print(id(word))
    properties.append('noun')
    print(id(properties))
    properties = 5
    print(id(properties))


w = ''
p = []
print(w, p)
set_up(w, p)
print(w, p, id(p))

from past.types import basestring


def tag(word):
    assert isinstance(word, basestring), "argument to tag() must be a string"
    if word in ['a', 'the', 'all']:
        return 'det'
    else:
        return 'noun'


print(tag(['ss']))


def accuracy(reference, test):
    """
    Calculate the fraction of test items that equal the corresponding reference items.
    Given a list of reference values and a corresponding list of test values,
    return the fraction of corresponding values that are equal.
    In particular, return the fraction of indexes
    {0<i<=len(test)} such that C{test[i] == reference[i]}.
    >>> accuracy(['ADJ', 'N', 'V', 'N'], ['N', 'N', 'V', 'ADJ'])
    0.5
    @param reference: An ordered list of reference values.
    @type reference: C{list}
    @param test: A list of values to compare against the corresponding
    reference values.
    @type test: C{list}
    @rtype: C{float}
    @raise ValueError: If C{reference} and C{length} do not have the
    same length.
    """
    if len(reference) != len(test):
        raise ValueError("Lists must have the same length.")
    num_correct = 0
    for x, y in izip(reference, test):
        if x == y:
            num_correct += 1
    return float(num_correct) / len(reference)


print(accuracy(['ADJ', 'N', 'V', 'N'], ['N', 'N', 'V', 'ADJ']))

sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the',
        'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']


def extract_property(prop):
    return [prop(word) for word in sent]


def last_letter(word):
    return word[-1]


print(extract_property(len))
print(extract_property(last_letter))
print(extract_property(lambda w: w[-1]))
print(sorted(sent, lambda x, y: cmp(len(y), len(x))))


def search1(substring, words):
    result = []
    for word in words:
        if substring in word:
            result.append(word)
    return result


def search2(substring, words):
    for word in words:
        if substring in word:
            yield word


print("search1:")
for item in search1('zz', nltk.corpus.brown.words()):
    print(item)
print("search2:")
for item in search2('zz', nltk.corpus.brown.words()):
    print(item)


def permutations(seq):
    if len(seq) <= 1:
        yield seq
    else:
        for perm in permutations(seq[1:]):
            for i in range(len(perm) + 1):
                yield perm[:i] + seq[0:1] + perm[i:]


for i in list(permutations(['police', 'fish', 'buffalo'])):
    print(i)


def is_content_word(word):
    return word.lower() not in ['a', 'of', 'the', 'and', 'will', ',', '.']


sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the',
        'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']
for i in filter(is_content_word, sent):
    print(i)

lengths = list(map(len, nltk.corpus.brown.sents(categories='news')))
# lengths = [len(w) for w in nltk.corpus.brown.sents(categories='news')]
print(sum(lengths) / len(lengths))

sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the',
        'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']
print(map(lambda w: len(filter(lambda c: c.lower() in "aeiou", w)), sent))
print([len([c for c in w if c.lower() in "aeiou"]) for w in sent])


def repeat(msg='<empty>', num=1):
    return msg * num


print(repeat(num=3))
print(repeat(msg='koo'))
print(repeat(msg='lml', num=3))


def generic(*args, **kwargs):
    print(args)
    print(kwargs)


print(generic(1, "African swallow", monty="python"))


def freq_words(file, min=1, num=10, verbose=False):
    freqdist = nltk.FreqDist()
    if trace: print("Opening", file)
    text = open(file).read()
    if trace: print("Read in %d characters" % len(file))
    for word in nltk.word_tokenize(text):
        if len(word) >= min:
            freqdist.inc(word)
            if trace and freqdist.N() % 100 == 0: print('.')
    if trace: print('\n')
    return freqdist.keys()[:num]


def size1(s):
    return 1 + sum(size1(child) for child in s.hyponyms())


def size2(s):
    layer = [s]
    total = 0
    while layer:
        total += len(layer)
        layer = [h for c in layer for h in c.hyponyms()]
    return total


dog = wn.synset('dog.n.01')
print(size1(dog), size2(dog))


def insert(trie, key, value):
    if key:
        first, rest = key[0], key[1:]
        if first not in trie:
            trie[first] = {}
        insert(trie[first], rest, value)
    else:
        trie['value'] = value


trie = nltk.defaultdict(dict)
insert(trie, 'chat', 'cat')
insert(trie, 'chien', 'dog')
insert(trie, 'chair', 'flesh')
insert(trie, 'chic', 'stylish')
trie = dict(trie)  # for nicer printing
trie['c']['h']['a']['t']['value']
print(trie)

from past.builtins import raw_input


def raw(file):
    contents = open(file).read()
    contents = re.sub(r'<.*?>', ' ', contents)
    contents = re.sub('\s+', ' ', contents)
    return contents


def snippet(doc, term):  # buggy
    text = ' ' * 30 + raw(doc) + ' ' * 30
    pos = text.index(term)
    return text[pos - 30:pos + 30]


print("Building Index...")
files = nltk.corpus.movie_reviews.abspaths()
idx = nltk.Index((w, f) for f in files for w in raw(f).split())
query = ''
while query != "quit":
    query = raw_input("query> ")
    if query in idx:
        for doc in idx[query]:
            print(snippet(doc, query))
    else:
        print("Not found")


def preprocess(tagged_corpus):
    words = set()
    tags = set()
    for sent in tagged_corpus:
        for word, tag in sent:
            words.add(word)
            tags.add(tag)
    wm = dict((w, i) for (i, w) in enumerate(words))
    tm = dict((t, i) for (i, t) in enumerate(tags))
    return [[(wm[w], tm[t]) for (w, t) in sent] for sent in tagged_corpus]


colors = 'rgbcmyk'  # red, green, blue, cyan, magenta, yellow, black


def bar_chart(categories, words, counts):
    "Plot a bar chart showing counts for each word by category"
    ind = pylab.arange(len(words))
    width = 1 / (len(categories) + 1)
    bar_groups = []
    for c in range(len(categories)):
        bars = pylab.bar(ind + c * width, counts[categories[c]], width,
                         color=colors[c % len(colors)])
        bar_groups.append(bars)
    pylab.xticks(ind + width, words)
    pylab.legend([b[0] for b in bar_groups], categories, loc='upper left')
    pylab.ylabel('Frequency')
    pylab.title('Frequency of Six Modal Verbs by Genre')
    pylab.show()


genres = ['news', 'religion', 'hobbies', 'government', 'adventure']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfdist = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in genres
    for word in nltk.corpus.brown.words(categories=genre)
    if word in modals)
counts = {}
for genre in genres:
    counts[genre] = [cfdist[genre][word] for word in modals]
bar_chart(genres, modals, counts)

import matplotlib

matplotlib.use('Agg')
pylab.savefig('modals.png')
print('Content-Type: text/html')
print('\n')
print('<html><body>')
print('<img src="modals.png"/>')
print('</body></html>')


def traverse(graph, start, node):
    graph.depth[node.name] = node.shortest_path_distance(start)
    for child in node.hyponyms():
        graph.add_edge(node.name, child.name)
        traverse(graph, start, child)


def hyponym_graph(start):
    G = nx.Graph()
    G.depth = {}
    traverse(G, start, start)
    return G


def graph_draw(graph):
    nx.draw(
        graph,
        node_size=[16 * graph.degree(n) for n in graph],
        node_color=[graph.depth[n] for n in graph],
        with_labels=False)
    matplotlib.pyplot.show()


dog = wn.synset('dog.n.01')
graph = hyponym_graph(dog)
graph_draw(graph)

import csv

input_file = open("lexicon.csv", "r")
for row in csv.reader(input_file):
    print(row)

a = np.array([[4, 0], [3, -5]])
u, s, vt = linalg.svd(a)

import nltk.cluster
