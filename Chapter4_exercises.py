from random import random
import matplotlib
import networkx as nx
import nltk
from nltk import ngrams, re
from nltk.corpus import state_union, swadesh, cmudict, wordnet as wn, shakespeare


# 1
help(str)
help(list)
help(tuple)

# 2
list1 = [1, 2, 3, 4]
tuple1 = (1, 'string', 3.1, [66, 1], ['dws', 'wqe'])
print(len(list1), len(tuple1))
print(list1[0], tuple1[0])
print(list1.count(1), tuple1.count(1))
print(list1.index(3), tuple1.index(1))
for i in list1:
    print(i)
for i in tuple1:
    print(i)
print(list1[-3:], tuple1[-3:])

list1.sort()
print(list1)
list1[1] = 'turned'
print(list1)
del list1[0]
print(list1)
tuple1.sort()
tuple1[1] = ('off', ['Qf', 'O:f'])
print(tuple1)
del tuple1[0]
print(tuple1)

list1 = list((2,'convnert'), (3, 5))
print(list1)

# 3
tuple1 = 'yuinbg',
print(type(tuple1), tuple1)
tuple1 = ('yubing', )
print(type(tuple1), tuple1)
tuple1 = tuple(('yubing', ))
print(type(tuple1), tuple1)

# 4
words = ['is', 'NLP', 'fun', '?']
tmp = words[0]
words[0] = words[1]
words[1] = tmp
words[3] = '!'
print(words)
words = ['is', 'NLP', 'fun', '?']
words[0], words[1], words[3] = 'NLP', 'is', '!'
print(words)

# 5 cmp is no longer available in python 3.6
help(cmp) # error

# 6
sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
for i in ngrams(sent, 3):
    print(i)
for i in ngrams(sent, 1):
    print(i)
for i in ngrams(sent, len(sent)):
    print(i)

# 7
string_empty = ''
list_empty = []
tuple_empty = ()
zero = 0
if not string_empty:
    print('False')
if not list_empty:
    print('False')
if not tuple_empty:
    print('False')
if zero == False:
    print('False')

# 8
print('Monty' < 'Python')
print('Z' < 'a')
print('z' < 'a')
print('Monty' < 'Montague')
print(('Monty', 1) < ('Monty', 2))
print(('Monty', 8, 'sw') < ('Monty', 2))

# 9
string = ' what can you do '
pattern = nltk.re.compile(r'^\s*|\s*$')
processed_str = pattern.sub('', string)
pattern = nltk.re.compile(r'\s')
processed_str = pattern.sub(' ', processed_str)
print(processed_str)

# 10 cmp is no longer available in python 3.6
def cmp_len(word1, word2):
    len1, len2 = len(word1), len(word2)
    if len1 != len2:
        return len(word1) < len(word2)
    else:
        return word1 < word2


wordlist = ['strng', 'fsa', 'fsc', 'gr', 'jyu', 'dejiaue']
# print(sorted(wordlist, cmp=cmp_len))
wordlist.sort(key=lambda x: len(x), reverse=True)
print(wordlist)

# 11
sent1 = ['strng', 'fsa', 'fsc', 'gr', 'jyu', 'dejiaue']
sent2 = sent1
sent1[0] = 'changed'
print(sent2)

sent1 = ['strng', 'fsa', 'fsc', 'gr', 'jyu', 'dejiaue']
sent2 = sent1[:]
sent1[0] = 'changed'
print(sent2)

text1 = [['string', 'fsa'], ['fsc', 'gr'], ['jyu', 'dejiaue']]
text2 = text1[:]
text1[0][0] = 'changed'
print(text2)

from copy import deepcopy
text1 = [['string', 'fsa'], ['fsc', 'gr'], ['jyu', 'dejiaue']]
text3 = deepcopy(text1)
text1[0][0] = 'changed'
print(text3)

# 12
word_table = [[''] * 2] * 5
word_table[0][1] = 'hello'
print(word_table)
word_table = []
for i in range(5):
    word_table.append(['']*2)
word_table[0][1] = 'hello'
print(word_table)

# 13
wordlist = ['strng', 'fsa', 'fsc', 'gr', 'jyu', 'dejiaue']
word_vowels = matrix = [[set() for i in range(8)] for i in range(8)]
for word in wordlist:
    l = len(word)
    cs = [c for c in word if c in 'aeiou']
    word_vowels[l][len(cs)].add(word)
for i in range(8):
    for j in range(8):
        print(word_vowels[i][j], end=' ')
    print('\n')


# 14
def novel10(text):
    cnt = int(len(text)*0.9)
    part1, part2 = text[:cnt], text[cnt:]
    rst = [w for w in part2 if w not in part1]
    return rst
raw = open('3_44.txt', encoding='utf8').read()
text = nltk.word_tokenize(raw)
print(text)
print(novel10(text))

# 15
sent = 'Increasingly the president raised doubts, without supportive evidence, about the integrity of the ' \
       'approaching November election.'
pattern = nltk.re.compile(r'[\,\.]')
lsent = pattern.sub('', sent)
lsent = lsent.split(' ')
print(lsent, len(lsent))
lsen_words = set([(w, lsent.count(w)) for w in lsent])
print(sorted(lsen_words, key=lambda x: x[0], reverse=False))

# 16
letter_vals = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 80, 'g': 3, 'h': 8,
               'i': 10, 'j': 10, 'k': 20, 'l': 30, 'm': 40, 'n': 50, 'o': 70, 'p': 80, 'q': 100,
               'r': 200, 's': 300, 't': 400, 'u': 6, 'v': 6, 'w': 800, 'x': 60, 'y': 10, 'z': 7}


def gematria(word):
    try:
        rst = [letter_vals[w] for w in word if w.isalpha()]
    except KeyError as key:
        return -1
    return sum(rst)

for fileid in state_union.fileids():
    text = state_union.words(fileid)
    text = [w.lower() for w in text]
    # print(text)
    words_666 = [w for w in text if gematria(w) == 666]
    print(len(words_666))

# ref https://github.com/walshbr/nltk/blob/master/ch_four/16.py
def decode(text):
    for i in range(1000):
        text_len = len(text)
        num = int(random() * text_len)
        text[num] = gematria(str(text[num]))
    print(text)
text = 'counts the number of words with a numerical count equal to ' * 1000
text = nltk.word_tokenize(text)
decode(text)

# 17
def common_words(text, n):
    text = [w.lower() for w in text if w.isalpha()]
    fdist = nltk.FreqDist(text)
    print(fdist.most_common(n))
    lst_com = [w for w, _ in fdist.most_common(n)]
    return lst_com
def shorten(text, n):
    com_words = common_words(text, n)
    rst = []
    for w in text:
        if w in com_words:
            rst.append('__')
        else:
            rst.append(w)
    rst = ' '.join(rst)
    return rst
text = state_union.words()
print(text)
print(shorten(text, 5))

# 18
def find_prononciations(word):
    prons = [pron for w, pron in cmudict.entries() if w == word]
    return prons
def find_meanings(word):
    means = [synset.definition() for synset in wn.synsets(word)]
    return means
def word_indexes(lexicon):
    index_list = []
    for word in lexicon:
        index_list.append((word, find_meanings(word), find_prononciations(word)))
    return index_list
lexicon = ['here', 'is', 'a', 'book']
index_list = word_indexes(lexicon)
for i in index_list:
    print(i)

# 19
def sort_synset(synset, lst):
    rst = sorted(lst, key=lambda x: wn.synset(x).path_similarity(synset))
    return rst
lst = ['minke_whale.n.01', 'orca.n.01', 'novel.n.01', 'tortoise.n.01']
print('right_whale.n.01', lst)

# 20
lst = ['open', 'table', 'chair', 'food', 'table', 'chair', 'table', 'chair', 'chair']
fdist = nltk.FreqDist(lst)
# print(lst)
slst = sorted(lst, key=lambda x: fdist[x])
print(sorted(set(slst), key=slst.index, reverse=True))

# 21
def difference(text, vocabulary):
    rst = [w for w in text if w not in vocabulary]
    return set(rst)
text = ['open', 'table', 'chair', 'food', 'table', 'chair', 'table', 'chair', 'chair']
vocabulary = ['chair']
print(difference(text, vocabulary))
print(set(text).difference(set(vocabulary)))

# 22
from operator import itemgetter
words = ['open', 'table', 'chair', 'food']
print(sorted(words, key=itemgetter(1))) # sort according to the second character of each word
print(sorted(words, key=itemgetter(-1))) # sort according to the last character of each word

# 23
def insert(trie, key, value):
    if key:
        first, rest = key[0], key[1:]
        if first not in trie:
            trie[first] = {}
        insert(trie[first], rest, value)
    else:
        trie['value'] = value

def lookup(trie, key):
    if key:
        first, rest = key[0], key[1:]
        if first not in trie:
            return 'Not such key!'
        value = lookup(trie[first], rest)
    else:
        try:
            value = trie['value']
        except KeyError as key:
            value = 'No such key!'
    return value


trie = nltk.defaultdict(dict)
insert(trie, 'vang', 'vanguard')
print(trie)
print(lookup(trie, 'vang'))

# 24 to be updated...
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

# 25 ref: https://www.jianshu.com/p/a617d20162cf
def Levenshtein_Distance_Recursive(str1, str2):
    if len(str1) == 0:
        return len(str2)
    elif len(str2) == 0:
        return len(str1)
    elif str1 == str2:
        return 0

    if str1[len(str1) - 1] == str2[len(str2) - 1]:
        d = 0
    else:
        d = 1

    return min(Levenshtein_Distance_Recursive(str1, str2[:-1]) + 1,
               Levenshtein_Distance_Recursive(str1[:-1], str2) + 1,
               Levenshtein_Distance_Recursive(str1[:-1], str2[:-1]) + d)


def Levenshtein_Distance_DP(str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if (str1[i - 1] == str2[j - 1]):
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


print(Levenshtein_Distance_Recursive("abc", "bd"))
print(Levenshtein_Distance_DP("abc", "bd"))

# 26
# ref: https://blog.csdn.net/L141210113/article/details/88075305
def catalan_re(n):
    if n == 1 or n == 0:
        return 1
    else:
        return (4 * n - 2) / (n + 1) * catalan_re(n - 1)

# ref: https://leetcode-cn.com/problems/unique-binary-search-trees/solution/qia-te-lan-shu-dong-tai-gui-hua-python3-by-zhu_shi/
def catalan_dp(n):
    if (n == 0):
        return 0
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - j - 1]
    return dp[-1]

print(int(catalan_re(3)))
print(catalan_dp(3))

from timeit import timeit
def test_re():
    return catalan_re(3)
def test_dp():
    return catalan_dp(3)
print(timeit(stmt=test_re, number=10))
print(timeit(stmt=test_dp, number=10))

