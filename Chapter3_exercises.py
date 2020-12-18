from __future__ import division
import random
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import brown, words, udhr, gutenberg, genesis, abc, wordnet as wn
from urllib.request import urlopen
import pycountry


# 1
string = 'colorness'
new_str = string[:4]+'u'+string[4:]
print(new_str)

# 2
string_list = ['dishes', 'running', 'nationality', 'undo', 'preheat']
rear_list = ['es', 'ning', 'ality', 'un', 'pre']
new_str_list = []
for s in string_list:
    target = False
    for r in rear_list:
        if r in s:
            target = True
            if s.index(r) == 0:
                new_str_list.append(s[len(r):])
            elif s.index(r) == len(s)-len(r):
                new_str_list.append(s[:-len(r)])
    if target == False:
        new_str_list.append(s)
print(new_str_list)

# 3
print('string'[-7])


# 4
string = "What a wonderful day!"
str1 = string[0:10:2]
str2 = string[-1:-10:-3]
print(str1, str2)


# 5
monty = 'one two three four five ... million'
print(monty[::-1])

# 6
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government. Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony. peat 1.82%"""
regs = [r'[a-zA-Z]+', r'[A-Z][a-z]*', r'p[aeiou]{,2}t', r'\d+(?:\.\d+)', r'([^aeiou][aeiou][^aeiou])*', r'\w+|[^\w\s]+']
for i in range(6):
    print('\n\n')
    nltk.re_show(regs[i], raw)

# 7
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government. Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony. an additional test"""
computation = "1+3*2 5*8 90+3 6*20"
print(nltk.re_show(r'\sa\s|\san\s|\s?(the)\s?', raw))
print(nltk.re.findall(r"\d+(?:[+*]\d+)*", computation))


# 8
html = urlopen('http://www.nltk.org/').read()
print(html[:60])
bs = BeautifulSoup(html, "html.parser")
raw = bs.get_text()
print(raw[:60])


# 9
def load(f):
    file = open(f).read()
    return file


text = load('corpus.txt')
pattern1 = r'''(?x) # set flag to allow verbose regexps
\?
| \,
| \;
| \"
| \'
| \(
| \)
| \:
| \-
| \_
| \`
| \!
| [\.]{3} # ellipsis
|\.
'''
pattern2 = r'''(?x) # set flag to allow verbose regexps
(?:[A-Z]\.)+ # abbreviations, e.g. U.S.A.
| \w+(?:[-']\w+)+ # words with optional internal hyphens
| (?:\$)\d+(?:\.\d+) # currency and percentages, e.g. $12.40
| [A-Z][a-z]+
'''
print(nltk.regexp_tokenize(text, pattern1))
print(nltk.regexp_tokenize(text, pattern2))


# 10
sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
result = []
for word in sent:
    result.append((word, len(word)))
print(result)


# 11
raw = "fcsavs vbajvio fcwashfi sfwejao wa ovsierg ebh o.f wefc ewa32rro v ds.fwef"
raw_sp = raw.split('s')
print(raw_sp)


# 12
string = 'string demo'
for i in string:
    print(i)


# 13
raw = "fcsavs\t   \t\t\t   fcwashfi sfwejao wa ovsierg ebh "
sp1 = raw.split()
sp2 = raw.split(' ')
print(sp1, sp2)


# 14 difference: whether to change words itself
words = ["fa", "fafc", "gre", "arfwe", "oge", 'pfges']
sorted(words)
print(words)
words.sort()
print(words)


# 15
string = "3" * 7
integer = 3 * 7
print(string, integer, int(string), str(integer))


# 16
import test
print(test.msg)
from test import msg
print(msg)


# 17
string1 = 'lml'
string2 = 'lml-lml'
print(".%6s and %-6s." %(string1, string1))
print(".%6s and %-6s." %(string2, string2))


# 18
text = open('corpus.txt').read()
pattern = r'[Ww][Hh]\w+'
print(sorted(nltk.re.findall(pattern, text)))
print(sorted(nltk.regexp_tokenize(text, pattern)))


# 19
text = open('3_19.txt').readlines()
print(text)
rst = []
for line in text:
    string, num = line.split(' ')
    num = int(num)
    rst.append([string, num])
print(rst)


# 20
url = "https://www.cnblogs.com/itdyb/p/5825860.html"
html = urlopen(url).read()
bs = BeautifulSoup(html, "html.parser")
raw = bs.get_text()
tokens = nltk.word_tokenize(raw)
text = nltk.Text(tokens)
print(text, text.collocations())


# 21
def unknown(url):
    html = urlopen(url).read()
    bs = BeautifulSoup(html, "html.parser")
    text = bs.get_text()
    print(text[:100])
    substring = nltk.re.findall(r'[a-z]+', text)
    rsts = [i for i in substring if i not in words.words()]
    print(rsts)
url = "https://www.furious.com/perfect/sina.html"
unknown(url)


# 22
def unknown(url):
    html = urlopen(url).read()
    bs = BeautifulSoup(html, "html.parser")
    text = bs.get_text()
    text = [w.lower() for w in text]
    print(text[:100])
    substring = nltk.re.findall(r'[a-z]+', str(text))
    rsts = [i for i in substring if i not in words.words()]
    print(rsts)
url = "http://global.chinadaily.com.cn/"
unknown(url)


# 23
print(nltk.re.findall(r"n't|\w+", "don't"))
print(nltk.re.findall(r"^(.*)(n't)$", "don't"))


# 24
raw = "seiols.ate"
change_list = {'e': '3', 'i': '1', 'o': '0', 'l': '|', 's': '5', '.': '5w33t!', 'ate': '8'}
for i in nltk.re.findall(r"ate|i|o|l|\.|s", raw):
    print(i)
    if i == 's':
        if raw.index(i) == 0 or raw[raw.index(i)-1] == ' ':
            index = raw.index(i)
            raw = list(raw)
            raw[index] = '$'
            raw = ''.join(raw)
        else:
            raw = raw.replace(i, change_list[i])
    else:
        raw = raw.replace(i, change_list[i])
for i in nltk.re.findall(r"e", raw):
    print(i)
    raw = raw.replace(i, change_list[i])
print(raw)


# 25
def convert_Pig_Latin(word):
    vowel = "AEIOUaeiou"
    for i in range(len(word)):
        if word[i] in vowel:
            new_str = word[i:] + word[:i] + 'ay'
            return new_str
print(convert_Pig_Latin("string"))

def convert_text(text):
    rst = []
    for word in text:
        rst.append(convert_Pig_Latin(word))
    return rst
text = "dcas fgaw gavw gvar grae quiet yellow happy style"
text = nltk.word_tokenize(text)
print(text, '\n', convert_text(text))

def convert_Pig_Latin_pro(word):
    vowel = "AEIOUaeiou"
    if 'y' in word:
        if word.index('y') < len(word) and word[word.index('y')+1] in vowel:
            pass
        else:
            vowel = vowel + 'y'
    for i in range(len(word)):
        if word[i] in vowel:
            new_str = word[i+1:] + word[:i+1] + 'ay'
            return new_str
print(convert_Pig_Latin_pro("quiet"), convert_Pig_Latin_pro("yellow"), convert_Pig_Latin_pro("style"))


# 26 ref: https://github.com/walshbr/nltk/blob/master/ch_three/26.py
def pull_out_vowels(word):
    """Takes in a word and pulls out all vowels for it."""
    word = word.lower()
    vowels = []
    for letter in word:
        if letter in "aeiou":
            vowels.extend(letter)
    vowels = nltk.bigrams(vowels)
    return vowels
def vowels_for_all_words(text):
    """pulls out all vowels for all words."""
    vowels = []
    for word in text:
        vowels.extend(pull_out_vowels(word))
    return vowels
text = udhr.words('Hungarian_Magyar-Latin1')
vowel_bigrams = vowels_for_all_words(text)
cfd = nltk.ConditionalFreqDist(vowel_bigrams)
cfd.tabulate()


# 27
word = []
for i in range(500):
    word.append(random.choice("aehh "))
print(word)
string = ''.join(word)
print(string)
word = string.split(' ')
print(word)
string = ''.join(word)
print(string)


# 29
def uw(words):
    letter_sum = 0
    for w in words:
        letter_sum += len(w)
    return letter_sum / len(words)


def us(sents):
    words_sum = 0
    for s in sents:
        words_sum += len(s)
    return words_sum / len(sents)


def ARI(uw, us):
    return 4.71 * uw + 0.5 * us - 21.43




words = brown.words(categories='lore')
print(words[0])
print(len(words))
sents = brown.sents(categories='lore')
print(sents[0])
print(len(sents))
print(ARI(uw(words), us(sents)))
words = brown.words(categories='learned')
sents = brown.sents(categories='learned')
print(ARI(uw(words), us(sents)))


# 30
raw = """Save some text into a file corpus.txt! Define a function load(f) that reads from
the file named in its sole argument, and returns a string containing the text of the
file. Can you hear me? He said,'Sure.' To-be-continued... $12.40, 2020-11-7.
state-of-the-art Monica. While you are alone, I am there. What a beautiful day!
Which do ypu like? We can go together."""
tokens = nltk.word_tokenize(raw)
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
print([porter.stem(t) for t in tokens])
print([lancaster.stem(t) for t in tokens])


# 31
saying = ['After', 'all', 'is', 'said', 'and', 'done', ',', 'more', 'is', 'said', 'than', 'done', '.']
lengths = []
for word in saying:
    lengths.append(len(word))
print(lengths)


# 32
silly = 'newly formed bland ideas are inexpressible in an infuriating way'
bland = silly.split(' ')
print(bland)
string = ''.join([s[1] for s in bland])
print(string)
string_original = ' '.join(bland)
print(string_original)
sorted_silly = sorted(bland)
for i in sorted_silly:
    print(i)


# 33
print('inexpressible'.index('re'))
words = ['newly', 'formed', 'bland', 'ideas', 'are', 'inexpressible', 'in', 'an', 'infuriating', 'way']
print(words.index('newly'))
silly = 'newly formed bland ideas are inexpressible in an infuriating way'
print(silly[:silly.index('in ')])


# 34 ref: https://github.com/walshbr/nltk/blob/master/ch_three/34.py#L28
def convert_nationality_adjectives(word):
    countries = [country.name for country in pycountry.countries]
    # list of regex things to check
    patterns = ['ese', 'ian', 'an', 'ean', 'n', 'ic', 'ern']
    # list of suffixes for appending to country names that get damaged when they are split.
    suffixes = ['a', 'o']
    for pattern in patterns:
        tup = nltk.re.findall(r'^(.*)(' + pattern + ')', word)
        if tup:
            country = tup[0][0]
            if country in countries:
                return country
            else:
                for suffix in suffixes:
                    new_country = country + suffix
                    if new_country in countries:
                        return new_country
    return "Not found!"
print(convert_nationality_adjectives('Mexican'))


# 35
raw = ' '.join(gutenberg.words('austen-emma.txt'))
print(nltk.re.findall(r'as best as \w+ can', raw))
print(nltk.re.findall(r'as best \w+ can', raw))


# 36 ref: https://github.com/walshbr/nltk/blob/master/ch_three/36.py
lolcat = genesis.words('lolcat.txt')
print(lolcat)
conversions = [['ight', 'iet'], ['i', 'ai'], ['y\s', 'eh '], ['he\s', 'him '], ['his\s', 'him '], ['she\s', 'her'],
               ['\shers\s', ' her'], ['they', 'dem'], ['their', 'dem'], ['y\s', 'eh'], ['th', 'f'], ['Th', 'F'],
               ['I\s', 'Ai '], ['I\sam', 'Iz'], ['me', 'meh'], ['you', 'yu'], ['them', 'dem'], ['le\s', 'el '],
               ['le\s', 'el '], ['ee', 'ea'], ['oa', 'ow'], ['er\s', 'ah']]

text = 'When I talk to you, you make certain assumptions about me as a person based on what you’re hearing. You ' \
       'decide whether or not I might be worth paying attention to, and you develop a sense of our social relations ' \
       'based around the sound of my voice. The voice conveys and generates assumptions about the body and about ' \
       'power: am I making myself heard? Am I registering as a speaking voice? Am I worth listening to? '

for c in conversions:
    old_letters = c[0]
    pattern = nltk.re.compile(r'(' + old_letters + ')')
    new_letters = c[1]
    text = pattern.sub(new_letters, text)

print(text)


# 37 ref: https://github.com/walshbr/nltk/blob/master/ch_three/37.py
raw = open('3_37_file.html', encoding='utf8').read()
# sets a pattern for stripping out tags
pattern = nltk.re.compile(r'<[^>]+>')
# strips them
processed_text = pattern.sub('', raw)
# sets a new pattern for normalizing whitespace.
pattern = nltk.re.compile(r'\s')
processed_text = pattern.sub(' ', processed_text)
print(processed_text)


# 38
text = "long-\nterm session pre-\ndefined"
print(text)
rst = nltk.re.findall(r'\w+-\n\w+', text)
print(rst)
pattern = nltk.re.compile(r'\n')
processed_text = pattern.sub('', text)
print(processed_text)


# 39
change_list = ['aeiouhwy', 'bfpv', 'cgjkqsxz', 'dt', 'l', 'mn', 'r']
def soundex(expression):
    num = []
    num.append(expression[0].lower())
    exp = expression[1:]
    for w in exp:
        for i in range(7):
            if i in num:
                pass
            elif w in change_list[i]:
                num.append(i)
    num.remove(0)
    string = ''
    for i in num[:4]:
        string += str(i)
    return string
print(soundex('Ahnddreg'))


# 40
words = abc.words('science.txt')
print(len(words))
sents = abc.sents('science.txt')
print(len(sents))
print(ARI(uw(words), us(sents)))
words = abc.words('rural.txt')
sents = abc.sents('rural.txt')
print(ARI(uw(words), us(sents)))
sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
text = abc.raw('science.txt')
sents = sent_tokenizer.tokenize(text)
print(sents[:18])


# 41
words = ['attribution', 'confabulation', 'elocution', 'sequoia', 'tenacious', 'unidirectional']
vsequences = set()
for word in words:
    vowels = word
    for char in vowels:
        if char not in 'aeiou':
            ind = vowels.index(char)
            vowels = vowels[:ind]+vowels[ind+1:]
    vsequences.add(''.join(vowels))
print(sorted(vsequences))


# 42 ref: https://github.com/walshbr/nltk/blob/master/ch_three/42.py
def sem_index(text):
    word_with_syns = []
    for word in text:
        synsets = wn.synsets(word)
        syns_indices = []
        for synset in synsets:
            # set the index number equal to its offset
            sem_index_num = synset.offset()
            syns_indices += [sem_index_num]
        if syns_indices:
            word_with_syns.extend((word, syns_indices))
        else:
            word_with_syns.extend((word, 'no synonyms'))
    return word_with_syns


text = genesis.words()
sem_index_nums = sem_index(text)
print(sem_index_nums[0:100])


# 43 ref: https://github.com/walshbr/nltk/blob/master/ch_three/43.py
def prep_mystery_text(text):
    """preps mystery text"""

    # pulls in the text whose language will be guessed.
    mystery_text = [list(word.lower()) for word in text if word.isalpha()]
    mystery_text = [item for sublist in mystery_text for item in sublist]
    fd_mystery_text = nltk.FreqDist(mystery_text)

    # pulls out a ranked list of characters based on the frequency distribution
    mystery_ranks = list(nltk.ranks_from_sequence(fd_mystery_text))

    return mystery_ranks


def prep_language_corpus(fids):
    # preps language corpus
    # pulls in all the languages, which udhr calls them the fileids)
    # fids = udhr.fileids()

    # makes a list of all the available languages that use Latin1 encoding.
    languages = [fileid for fileid in fids if nltk.re.findall('Latin1', fileid)]

    # pulls in all of the udhr for all diff. languages broken apart by characters.

    udhr_corpus = [[list(word.lower()) for word in udhr.words(language) if word.isalpha()] for language in languages]

    # flattens that list so that it is a clump of letters for each language

    udhr_corpus = [[item for sublist in language for item in sublist] for language in udhr_corpus]

    # gives the languages all indices. So you can pull in the text of the udhr by knowing its index number a la
    # udhr_corpus[154] returns spanish

    languages = list(enumerate(languages))

    # gets frequency distributions for all the characters in a list. then converts it to a ranked list

    language_freq_dists = [nltk.FreqDist(language) for language in udhr_corpus]
    language_ranks = [list(nltk.ranks_from_sequence(dist)) for dist in language_freq_dists]

    return languages, language_ranks


def spearman(mystery_ranks, language_ranks):
    """spearman correlation bit. compares the ranks for the mystery text with the ranks of every other language
"""
    spearman_numbers = []
    for language in language_ranks:
        number = nltk.spearman_correlation(language, mystery_ranks)
        spearman_numbers.append(number)

    return spearman_numbers


def calculate(text, fids):
    """zips the spearman correlation numbers into a single list along with the language list and their indices."""

    languages, language_ranks = prep_language_corpus(fids)
    mystery_ranks = prep_mystery_text(text)
    spearman_numbers = spearman(mystery_ranks, language_ranks)
    zipped = list(zip(languages, spearman_numbers))

    # sorts it all by the spearman correlation, and then pops the last one (highest one) off and prints it out.
    # That's the computer's best guess as to what is the same.

    zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    return zipped


if __name__ == '__main__':
    fids = ['French_Francais-Latin1', 'Spanish-Latin1', 'German_Deutsch-Latin1', 'English-Latin1']
    # fids = list(udhr.fileids())
    text = gutenberg.words('austen-emma.txt')
    answer = calculate(text, fids)
    print(answer)


# 44
def similarity(word1, word2):
    # print(word1, word2)
    sn1 = wn.synsets(word1)
    # print(sn1)
    sn2 = wn.synsets(word2)
    max_sim = 0.0
    for s1 in sn1:
        # print('\n', s1)
        for s2 in sn2:
            # print(s2)
            cur_sim = s1.path_similarity(s2)
            # print(cur_sim)
            try:
                if max_sim < cur_sim:
                    max_sim = cur_sim
            except TypeError:
                pass
        # print(max_sim)
    return max_sim


word1 = 'booming'
text = open('3_44.txt', encoding='utf8').read()
text = nltk.word_tokenize(text)
text = set([word.lower() for word in text if word.lower() != word1 and word.isalpha()])
print(text)
pairs = [(word1, word) for word in text]
print(len(pairs))
sim_list = [similarity(word1, word2) for word1, word2 in pairs]
# print(sim_list[:20])
sorted_pairs = sorted(pairs, key=lambda x: similarity(x[0], x[1]), reverse=True)
sorted_sim = [similarity(p[0], p[1]) for p in sorted_pairs]
print(sorted_pairs[:20], '\n', sorted_sim[:20])


# 45
#No solution...

