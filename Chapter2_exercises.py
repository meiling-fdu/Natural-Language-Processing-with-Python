from __future__ import division
import nltk
from nltk.corpus import gutenberg, brown, state_union, swadesh, names, cmudict, udhr
from nltk.corpus import wordnet as wn


# 1
phrase = ["I", "like", 'noodles', "."]
print(phrase+phrase)
print(phrase*3)
print(phrase[2])
print(phrase[:3])
print(sorted(phrase))

# 2
text = gutenberg.words('austen-persuasion.txt')
print('word token: ', len(text))
print('word type: ', len(set(text)))

# 3
print(brown.categories())
print(brown.words(categories='fiction'))
print(brown.words(categories='humor'))

# 4
su = state_union.words()
cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in state_union.fileids()
    for word in [w.lower() for w in state_union.words(fileid)]
    for target in ['men', 'women', 'people']
    if target == word
)
cfd.plot()

# 5
print(wn.synsets('rabbit'))
print(wn.synset('rabbit.n.01').member_meronyms())
print(wn.synset('rabbit.n.01').part_meronyms())
print(wn.synset('rabbit.n.01').substance_meronyms())
print(wn.synset('rabbit.n.01').member_holonyms())
print(wn.synset('rabbit.n.01').part_holonyms())
print(wn.synset('rabbit.n.01').substance_holonyms())
paths = wn.synset('rabbit.n.01').hypernym_paths()
print(len(paths), paths[0])

print(wn.synsets('pig'))
print(wn.synset('hog.n.01').member_meronyms())
print(wn.synset('hog.n.01').part_meronyms())
print(wn.synset('hog.n.01').substance_meronyms())
print(wn.synset('hog.n.01').member_holonyms())
print(wn.synset('hog.n.01').part_holonyms())
print(wn.synset('hog.n.01').substance_holonyms())
paths = wn.synset('hog.n.01').hypernym_paths()
print(len(paths), paths[0], paths[1])

print(wn.synset('rabbit.n.01').lowest_common_hypernyms(wn.synset('hog.n.01')))

# 6
print(swadesh.fileids())
print(swadesh.words('en'))
en2fr = swadesh.entries(['en', 'fr'])
translate = dict(en2fr)
mydict = dict([('rabbit', 'add_lml')])
translate.update(mydict)
print(translate['pig'])
try:
    print(translate['pig'])
except KeyError as key:
    print('No match!')

# 7
emma = nltk.Text(gutenberg.words('austen-emma.txt'))
print(emma.concordance('However', width=50, lines=5))
print(emma.similar('However'))

# 8
cfd = nltk.ConditionalFreqDist(
    (fileid[:-4], name[0])
    for fileid in names.fileids()
    for name in names.words(fileid)
)
cfd.plot()

# 9
from nltk.book import text1, text2
print(text1.concordance('love', width=50, lines=5))
print(text2.concordance('love', width=50, lines=5))
print(text1.similar('love'))
print(text2.similar('love'))

# 10
freq1 = nltk.FreqDist(text1)
freq1.plot(50)
prop = len(text1)/3
print([w for w in set(text1) if freq1[w] > prop])

# 11
type = ['news', 'romance']
emotions = ['like', 'dislike', 'hate', 'love', 'ignore', 'curse', 'enjoy']
cfd = nltk.ConditionalFreqDist(
    (genre, emotion)
    for genre in type
    for word in brown.words(categories=genre)
    for emotion in emotions
    if emotion == word
)
cfd.plot()

# 12
entries = cmudict.entries()
print(len(entries))
wordlist = set([word for word, pron in entries])
print(len(wordlist))
totalword = [word for word, pron in entries]
oneprop = [word for word in wordlist if totalword.count(word) == 1]
print(len(oneprop))
print(len(oneprop)/len(wordlist))

cfd = nltk.ConditionalFreqDist(
    (w, totalword.count(w)/len(totalword))
    for w in totalword
)
print(cfd.tabulate())

# 13
# pay attention to difference between generator and list
noun = wn.all_synsets('n')
noun_num = len(list(noun))
lst = [ss for ss in wn.all_synsets('n') if len(ss.hyponyms()) <= 0]   # wn.all_synsets('n') can not be noun!
noun_no_hypo = len(list(lst))
print(noun_num, noun_no_hypo, noun_no_hypo / noun_num)

# 14
def supergloss(synset):
    str = ''
    hypernyms = synset.hypernyms()
    for hyper in hypernyms:
        str += hyper.definition() + '\n'
    str += '\n\n' + synset.definition() + '\n\n\n'
    hyponyms = synset.hyponyms()
    for hypo in hyponyms:
        str += hypo.definition() + '\n'
    print(str)
supergloss(wn.synset('car.n.01'))

# 15
fdist = nltk.FreqDist(brown.words())
lst = [words for words in brown.words() if fdist[words] >= 3]
print(len(lst), lst[:5])

# 16
def tokens(text):
    return len(text)

def types(text):
    return len(set(text))

def lexical_diversity(tokens_um, types_num):
    return tokens_um / types_num

import prettytable as pt

tb = pt.PrettyTable(["Genre", "Tokens", "Types", "Lexical diversity"])

for genre in brown.categories():
    text = brown.words(categories=genre)
    tok = tokens(text)
    typ = types(text)
    div = lexical_diversity(tok, typ)
    tb.add_row([genre, tok, typ, round(div, 1)])
print(tb)

# 17
from nltk.corpus import stopwords
def most_common(text):
    fdist1 = nltk.FreqDist([w for w in text if w not in stopwords.words('english')])
    print(fdist1.most_common(50))
    fdist1.plot(50)
most_common(gutenberg.words('austen-emma.txt'))

# 18
text = gutenberg.words('austen-emma.txt')
text = [w for w in text if w not in stopwords.words('english')]
bigrams = nltk.bigrams(text)
fdist = nltk.FreqDist(bigrams)
print(fdist.most_common(50))
fdist.plot(50)

# 19 similar as 11
emotions = ['like', 'hate', 'love', 'mad', 'fear', 'sad', 'happy']
cfd = nltk.ConditionalFreqDist(
    (genre, emotion)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
    for emotion in emotions
    if emotion == word
)
cfd.plot()
cfd.tabulate()

# 20
def word_freq(word, section):
    text = section.words()
    print(100 * text.count(word)/len(text))
word_freq('like', brown)

# 21
def syllables_num(text):
    entries = cmudict.entries()
    prons = [pron for word, pron in entries if word in text]
    syllables = [syls for syls in prons]
    return len(syllables)


ss = set([w for w in gutenberg.words('austen-emma.txt') if w.startswith("ant")])
print(syllables_num(gutenberg.words('austen-emma.txt')))

# 22
import numpy as np

def hedge(text):
    new_version = []
    i = 0
    for words in text:
        new_version.append(words)
        i += 1
        if i % 3 == 0:
            new_version.append("like")
    # another method
    # i = 0
    # while (i+3) <= len(text):
    #     new_version = np.concatenate((new_version, text[i:i+3]))
    #     new_version = np.concatenate((new_version, ["like"]))
    #     i += 3
    # if i < len(text):
    #     new_version = np.concatenate((new_version, text[i:]))

    return new_version

print(hedge(['I', 'love', 'rabbit', 'and', 'peggy']))

# 23
import random
import re
import matplotlib.pyplot as plt


def zipf(text):
    fdist = nltk.FreqDist([w.lower() for w in text if w.isalpha()])
    # print(fdist.keys())
    sorted_brown_words = sorted(fdist.keys(), key=lambda x: fdist[x], reverse=True)
    outfile = open(r"2_23_1.txt", 'w')
    for w in sorted_brown_words:
        outfile.write("%s\t%d\n" % (w, fdist[w]))
        plt.scatter(w, fdist[w])
    plt.show()


zipf(brown.words())

text = ""
for i in range(500000):
    text += random.choice("abcdefg ")
word_li = re.split(r"\s+", text)  # according to space to split
fdist = nltk.FreqDist(word_li)
sorted_word_li = sorted(fdist.keys(), key=lambda x: fdist[x], reverse=True)
with open(r'\2_23_2.txt', 'w') as outfile:
    for w in sorted_word_li:
        outfile.write("%s\t%d\n" % (w, fdist[w]))

# 24
import random
def word_selection(n, text):
    fdist = nltk.FreqDist(text)
    return(fdist.most_common(n))
def generate_model(text, word, num=20):
    bigrams = nltk.bigrams(text)
    cfd = nltk.ConditionalFreqDist(bigrams)
    for i in range(num):
        print(word, end=' ')
        word = cfd[word].max()

text = brown.words(categories='fiction')
words = [w for w, num in word_selection(n=10, text=text)]
print(words)
word = random.choice(words)
print(word)
generate_model(text, word, 50)

# 25
def find_language(string):
    languages = [lang for lang in udhr.fileids() if '-Latin1' in lang]
    rst = [lang for lang in languages if string in udhr.words(lang)]
    return rst
print(find_language('I'))

# 26
branch_sum = 0
synset_num = len(list(wn.all_synsets('n')))
branch_list = []
print(synset_num)
for synset in wn.all_synsets('n'):
    branch_num = len(list(synset.hyponyms()))
    branch_list.append(branch_num)
    branch_sum += branch_num
print(branch_sum, synset_num, branch_sum/synset_num)
print(branch_list)
fdist = nltk.FreqDist(branch_list)
fdist.plot()

# 27
noun_set = set([x.name().split(".")[0] for x in wn.all_synsets('n')])
noun_num = len(noun_set)
sem_sum = 0
sem_list = []
for word in noun_set:
    sem_num = len(wn.synsets(word, 'n'))
    sem_sum += sem_num
    sem_list.append(sem_num)
print(sem_sum, noun_num, sem_sum/noun_num)
fdist = nltk.FreqDist(sem_list)
fdist.plot()

# 28
# Hint: the similarity of a pair should be represented by the similarity of the most similar pair of synsets they have.

def similarity(pair):
    word1, word2 = pair.split('-')
    print(word1, word2)
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


pairs = ["car-automobile", "gem-jewel", "journey-voyage", "boy-lad", "coast-shore", "asylum-madhouse",
         "magician-wizard", "midday-noon", "furnace-stove", "food-fruit", "bird-cock", "bird-crane",
         "tool-implement", "brother-monk", "lad-brother", "crane-implement", "journey-car", "monk-oracle",
         "cemetery-woodland", "food-rooster", "coast-hill", "forest-graveyard", "shore-woodland", "monk-slave",
         "coast-forest", "lad-wizard", "chord-smile", "glass-magician", "rooster-voyage", "noon-string"]
sim_list = [similarity(p) for p in pairs]
print(sim_list)
sorted_pairs = sorted(pairs, key=lambda x: similarity(x), reverse=True)
print(sorted_pairs)
