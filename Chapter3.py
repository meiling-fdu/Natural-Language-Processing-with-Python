from __future__ import division
import nltk, re, pprint
from nltk.corpus import gutenberg, nps_chat, brown
from urllib.request import urlopen
from bs4 import BeautifulSoup



# access electronic books
url = "http://www.gutenberg.org/files/2358/2358.txt"
raw = urlopen(url).read()
raw = str(raw)
print(type(raw), len(raw), raw[:8])


# tokenization
tokens = nltk.word_tokenize(raw)
print(type(tokens), len(tokens), tokens[:8])

text = nltk.Text(tokens)
print(type(text), len(text))
print(text.collocations())

print(raw.find("Chapter I"))
print(raw.rfind("End of Project Gutenberg's Crime"))  #reverse find
raw = raw[5303:1157681]
print(raw.find("Chapter I"))


url = "https://www.cnblogs.com/itdyb/p/5825860.html"
html = urlopen(url).read()
print(html[:60])

bs = BeautifulSoup(html, "html.parser")
raw = bs.get_text()
tokens = nltk.word_tokenize(raw)
text = nltk.Text(tokens)
print(text, text.collocations())


# RSS
import feedparser
llog = feedparser.parse('http://feed.cnblogs.com/blog/u/161528/rss')
print(llog['feed']['title'])
print(len(llog.entries))
post = llog.entries[2]
print(post.title)
content = post.content[0].value
print(type(content), content[:70])
bs = BeautifulSoup(content, features='lxml')
raw = bs.get_text()
token = nltk.word_tokenize(content)
print(token)

# open local files
f = open("chapter3.txt")
raw = f.read()
print(raw)
import os
print(os.listdir('.'))

path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')
raw = open(path).read()
print(len(raw))


# interact with users
from past.builtins import raw_input
s = raw_input("Enter some words: ")
print("You typed", len(nltk.word_tokenize(s)), "words.")


# string operations
couplet = '''Rough winds do shake the darling buds of May,
----------
And Summer's lease hath all too short a date:'''
print(couplet)


a = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
b = [' ' * 2 * (7 - i) + 'very' * i for i in a]
for line in b:
    print(b)

str = "string example"
str[0]='l'

nacute = u'\u0144'
nacute_utf = nacute.encode('utf8')
print(repr(nacute_utf))
print(nacute, nacute_utf)


# regular expressions
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

print([w for w in wordlist if re.search('ed$', w)])
print([w for w in wordlist if re.search('^..j..t..$', w)])
print([w for w in wordlist if re.search('..j..t..$', w)])
# wordlist = ['email', 'e-mail']
print([w for w in wordlist if re.search('^e-?mail$', w)])
print([w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)])
print([w for w in wordlist if re.search('^[g-o]$', w)])
print([w for w in wordlist if re.search('^[a-fj-o]+$', w)])

chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
print([w for w in chat_words if re.search('^m+i+n+e+$', w)])
print([w for w in chat_words if re.search('^m*i*n*e*$', w)])
print([w for w in chat_words if re.search('^[^aeiouAEIOU]+$', w)])

wsj = sorted(set(nltk.corpus.treebank.words()))
print([w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)])
print([w for w in wsj if re.search('^[A-Z]+\$$', w)])
print([w for w in wsj if re.search('^[0-9]{4}$', w)])
print([w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)])
print([w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)])
print([w for w in wsj if re.search('(ed|ing)$', w)])
print([w for w in wsj if re.search('ed|ing$', w)])  # words that contain ed or end with ing
print([w for w in wsj if re.search('t(ed|ing)+$', w)])

word = 'supercalifragilisticexpialidocious'
print(re.findall(r'[aeiou]', word))
fd = nltk.FreqDist(vs for word in wsj
                   for vs in re.findall(r'[aeiou]{2,}', word)
                   )
print(fd.items())

print([int(n) for n in re.findall('[0-9]{4}|[0-9]{2}', '2009-12-31')])

regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
def compress(word):
    pieces = re.findall(regexp, word)
    return ''.join(pieces)
english_udhr = nltk.corpus.udhr.words('English-Latin1')
print(nltk.tokenwrap(compress(w) for w in english_udhr[:30]))

rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
cfd = nltk.ConditionalFreqDist(cvs)
cfd.tabulate()

cv_word_pairs = [(cv, w) for w in rotokas_words
                 for cv in re.findall(r'[ptksvr][aeiou]', w)]
cv_index = nltk.Index(cv_word_pairs)
cv_index['su']

print(re.findall(r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing'))
print(re.findall(r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing'))
print(re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing'))
print(re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processes'))
print(re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$', 'language'))

def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government. Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = nltk.word_tokenize(raw)
print([stem(t) for t in tokens])

moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
print(moby.findall(r"<a> (<.*>) <man>"))
chat = nltk.Text(nps_chat.words())
print(chat.findall(r"<.*> <.*> <bro>"))
print(chat.findall(r"<l.*>{3,}"))

print(nltk.re_show(r"<love><[b-o]*>$", 'love bobo'))
nltk.app.nemo()

hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))
print(hobbies_learned.findall(r"<\w*> <and> <other> <\w*s>"))
print(hobbies_learned.findall(r"<as> <\w*> <as> <\w*>"))

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
    is no basis for a system of government. Supreme executive power derives from
    a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = nltk.word_tokenize(raw)
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
print([porter.stem(t) for t in tokens])
print([lancaster.stem(t) for t in tokens])


class IndexedText(object):
    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                         for (i, word) in enumerate(text))
    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = width//4 # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '%*s' % (width, lcontext[-width:])
            rdisplay = '%-*s' % (width, rcontext[:width])
            print(ldisplay, rdisplay)
    def _stem(self, word):
        return self._stemmer.stem(word).lower()
porter = nltk.PorterStemmer()
grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter, grail)
print(text.concordance('lie'))


wnl = nltk.WordNetLemmatizer()
print([wnl.lemmatize(t) for t in tokens])


raw = """DENNIS: Listen, strange women lying in ponds distributing swords
    is no basis for a system of government. Supreme executive power derives from
    a mandate from the masses, not from some farcical aquatic ceremony."""
print(re.split(r' ', raw))
print(re.split(r'[ \t\n]+', raw))
print(re.split(r'\s+', raw))
print(re.split(r'\W+', raw))
print('xx'.split('x'))
print(re.findall(r'\w+|\S\w*', raw))
print(re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw))


text = r"That U.S.A. poster-print costs $12.40..."
pattern = r'''(?x) # set flag to allow verbose regexps
(?:[A-Z]\.)+ # abbreviations, e.g. U.S.A.
| \w+(?:[-']\w+)* # words with optional internal hyphens
| (?:\$)\d+(?:\.\d+) # currency and percentages, e.g. $12.40, 82%
| [\.]{3} # ellipsis
| (?:[.,;"'?():-_`]) # these are separate tokens
'''
print(nltk.regexp_tokenize(text, pattern))


nltk.corpus.treebank_raw.raw()
nltk.corpus.treebank.words()


print(len(brown.words()) / len(brown.sents()))


sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
text = gutenberg.raw('chesterton-thursday.txt')
sents = sent_tokenizer.tokenize(text)
print(sents[171:181])


def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words
text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"
print(segment(text, seg1))
print(segment(text, seg2))


def evaluate(text, segs):
    words = segment(text, segs)
    text_size = len(words)
    lexicon_size = len(' '.join(list(set(words))))
    return text_size + lexicon_size
text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"
seg3 = "0000100100000011001000000110000100010000001100010000001"
evaluate(text, seg3)


from random import randint
def flip(segs, pos):
    return segs[:pos] + str(1-int(segs[pos])) + segs[pos+1:]
def flip_n(segs, n):
    for i in range(n):
        segs = flip(segs, randint(0,len(segs)-1))
    return segs
def anneal(text, segs, iterations, cooling_rate):
    temperature = float(len(segs))
    while temperature > 0.5:
        best_segs, best = segs, evaluate(text, segs)
        for i in range(iterations):
            guess = flip_n(segs, int(round(temperature)))
            score = evaluate(text, guess)
            if score < best:
                 best, best_segs = score, guess
        score, segs = best, best_segs
        temperature = temperature / cooling_rate
        print(evaluate(text, segs), segment(text, segs))
text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
anneal(text, seg1, 5000, 1.2)


silly = ['We', 'called', 'him', 'Tortoise', 'because', 'he', 'taught', 'us', '.']
print(' '.join(silly))


def tabulate(cfdist, words, categories):
    print('%-16s' % 'Category', end=' ')
    for word in words:  # column headings
        print('%6s' % word, end=' ')
    print('\n')
    for category in categories:
        print('%-16s' % category, end=' ')  # row heading
        for word in words:  # for each word
            print('%6d' % cfdist[category][word], end=' ')  # print table cell
        print('\n')  # end the row
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
tabulate(cfd, modals, genres)


output_file = open('chaper3_output.txt', 'w')
words = set(nltk.corpus.genesis.words('english-kjv.txt'))
for word in sorted(words):
    output_file.write(word + "\n")
output_file.close()


saying = ['After', 'all', 'is', 'said', 'and', 'done', ',', 'more', 'is', 'said', 'than', 'done', '.']
from textwrap import fill
format = '%s (%d),'
pieces = [format % (word, len(word)) for word in saying]
output = ' '.join(pieces)
wrapped = fill(output)
print(wrapped)

