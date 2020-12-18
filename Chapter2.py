import nltk
from nltk.corpus import gutenberg, webtext, nps_chat, brown, reuters, inaugural, udhr, stopwords



print(nltk.corpus.gutenberg.fileids())

emma = nltk.corpus.gutenberg.words('austen-emma.txt')
print(emma, len(emma))

emma = nltk.Text(gutenberg.words('austen-emma.txt'))
print(emma.concordance('surprize'))

for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
    print(num_chars/num_words, num_words/num_sents, num_words/num_vocab, fileid)

print(gutenberg.raw('austen-emma.txt'))
print(gutenberg.words('austen-emma.txt'))
print(gutenberg.sents('austen-emma.txt'))

for fileid in webtext.fileids():
    num_chars = len(webtext.raw(fileid))
    num_words = len(webtext.words(fileid))
    num_sents = len(webtext.sents(fileid))
    num_vocab = len(set([w.lower() for w in webtext.words(fileid)]))
    print(num_chars/num_words, num_words/num_sents, num_words/num_vocab, fileid)

print(webtext.raw('singles.txt'))
print(webtext.words('singles.txt'))
print(webtext.sents('singles.txt'))

chatroom = nps_chat.posts('10-19-20s_706posts.xml')
print(chatroom[123])

print(brown.categories())
print(brown.raw(fileids=['cg22']))
print(brown.words(categories='news'))
print(brown.words(fileids=['cg22']))
print(brown.sents(fileids=['cg22']))

news_text = brown.words(categories='news')
fdist = nltk.FreqDist([w.lower() for w in news_text])
modals = ['might', 'may', 'could', 'can', 'must', 'will']
for m in modals:
    print(m, ': ', fdist[m])

humor_text = brown.words(categories='humor')
fdist = nltk.FreqDist([w.lower() for w in humor_text])
lst = set(w.lower() for w in humor_text if w.startswith('wh'))
print(lst)
for wh in lst:
    print(wh, ': ', fdist[wh])

cfc = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)
genres = ['fiction', 'government', 'hobbies', 'humor', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']
modals = ['might', 'may', 'could', 'can', 'must', 'will']
print(cfc.tabulate(conditions=genres, samples=modals))

print(reuters.fileids(['acq', 'barley']))
print(reuters.categories(['training/9865']))
print(reuters.words(['training/9865']))

print(inaugural.fileids())
cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower() .startswith(target)
)
cfd.plot()


print(udhr.fileids())
languages = ['Chickasaw', 'English', 'German_Deutsch','Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)

raw_text = udhr.raw('Achehnese-Latin1')
nltk.FreqDist(raw_text).plot()

print(udhr.abspath('Achehnese-Latin1'), udhr.encoding('Achehnese-Latin1'), udhr.open('Achehnese-Latin1'))
print(udhr.root(), udhr.readme())

from nltk.corpus import PlaintextCorpusReader

corpus_root = 'E:\Learning Books'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
print(wordlists.fileids())
# wordlists.words('connectives')


from nltk.corpus import BracketParseCorpusReader

corpus_root = r"E:\nltk_data\corpora\gutenberg"
file_pattern = r".*"
ptb = BracketParseCorpusReader(corpus_root, file_pattern)
print(ptb.fileids())
print(len(ptb.fileids()))
print(ptb.sents(fileids='whitman-leaves.txt')[:19])


# Conditional Frequency Distribution

cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in ['news', 'romance']
    for word in brown.words(categories=genre)
)
print(cfd.conditions())
print(cfd['news'], cfd['romance'])
print(list(cfd['romance'])[:20])
print(cfd['romance']['could'])


cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower() .startswith(target)
)
cfd.plot()


languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1')
)
cfd.plot(conditions=['English', 'German_Deutsch'], samples=range(10), cumulative=False)
cfd.tabulate(conditions=['English', 'German_Deutsch'], samples=range(15), cumulative=True)


days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
cfd = nltk.ConditionalFreqDist(
    (genre, day)
    for genre in ['news', 'romance']
    for day in brown.words(categories=genre)
    if day in days
)
cfd.plot()


# example 2.1
def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word)
        word = cfdist[word].max()


text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)
print(list(cfd['living'].most_common()))
generate_model(cfd, 'many', 20)

cfd.plot(conditions=['living'])
print(cfd[2][2])


# example 2.3
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)


print(unusual_words(nltk.corpus.nps_chat.words()))


from nltk.corpus import stopwords
print(sorted(stopwords.words('english')))


def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)


print(content_fraction(nltk.corpus.reuters.words()))


puzzle_letters = nltk.FreqDist('egivrvonl')
print(puzzle_letters.plot())
obligatory = 'r'
wordlist = nltk.corpus.words.words()
print([w for w in wordlist if len(w) >= 6 and obligatory in w and nltk.FreqDist(w) <= puzzle_letters])


names = nltk.corpus.names
print(list(names.fileids()))
male_names = names.words('male.txt')
female_names = names.words('female.txt')
print([w for w in female_names if w in male_names])


names = nltk.corpus.names
cfd = nltk.ConditionalFreqDist(
    (fileid, name[-1])
    for fileid in names.fileids()
    for name in names.words(fileid)
)
cfd.plot()


entries = nltk.corpus.cmudict.entries()
print(len(entries))
print([entry for entry in entries[:6]])

for word, pron in entries:
    if len(pron) == 3:
        ph1, ph2, ph3 = pron
        if ph1 == 'P' and ph3 == 'T':
            print(word, ph2)

syllable = ['N', 'IH0', 'K', 'S']
print([word for word, pron in entries if pron[-4:] == syllable])

print([w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n'])
print(sorted(set(w[:2] for w, pron in entries if pron[0] == 'N' and w[0] != 'n')))


def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()]


print([w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']])


p3 = [(pron[0]+'-'+pron[2], word) for (word, pron) in entries if pron[0] == 'P' and len(pron) == 3]
cfd = nltk.ConditionalFreqDist(p3)
for template in cfd.conditions():
    if len(cfd[template]) > 10:
        words = cfd[template].keys()
        wordlist = ' '.join(words)
        print(template, wordlist[:70] + "...")


prondict = nltk.corpus.cmudict.dict()
print(prondict['fire'])
text = ['natural', 'language', 'processing']
print([ph for w in text for ph in prondict[w][0]])


from nltk.corpus import swadesh

print(swadesh.fileids())
print(swadesh.words('en'))
fr2en = swadesh.entries(['fr', 'en'])  # French-English
translate = dict(fr2en)
print(translate['chien'])

de2en = swadesh.entries(['de', 'en'])  # German-English
translate.update(dict(de2en))
print(translate['Hund'])


languages = ['en', 'de', 'nl', 'es', 'fr', 'pt', 'la']
for i in [139, 140, 141, 142]:
    print(swadesh.entries(languages)[i])


from nltk.corpus import toolbox
print(toolbox.entries('rotokas.dic'))


from nltk.corpus import wordnet as wn
print(wn.synsets('motorcar'))
print(wn.synset('car.n.01').lemma_names())
print(wn.synset('car.n.01').definition())
print(wn.synset('car.n.01').examples())
print(wn.synset('car.n.01').lemmas())
print(wn.lemma('car.n.01.automobile').synset())
print(wn.lemma('car.n.01.automobile').name())
print(wn.synsets('car'))
for synset in wn.synsets('car'):
    print(synset.lemma_names())
print(wn.lemmas('car'))



print(wn.synsets('dish'))
for synset in wn.synsets('dish'):
    print(synset.lemmas())
    print(synset.definition())
    print(synset.examples())


motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
print(types_of_motorcar[26])
print(sorted([lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas()]))


print(motorcar.hypernyms())
paths = motorcar.hypernym_paths()
print(len(paths))
print([synset.name() for synset in paths[0]])
print(motorcar.root_hypernyms())


nltk.app.wordnet()


print(wn.synset('tree.n.01').part_meronyms())
print(wn.synset('tree.n.01').substance_meronyms())
print(wn.synset('tree.n.01').member_holonyms())


for synset in wn.synsets('mint', wn.NOUN):
    print(synset.name() + ':', synset.definition())
print(wn.synset('mint.n.04').part_holonyms())
print(wn.synset('mint.n.04').substance_holonyms())


mint_4 = wn.synset('mint.n.04')
paths = mint_4.hypernym_paths()
print(len(paths))
print([synset.name() for synset in paths[0]])
print(mint_4.root_hypernyms())


print(wn.synset('walk.v.01').entailments())
print(wn.synset('eat.v.01').entailments())
print(wn.lemma('supply.n.02.supply').antonyms())
print(wn.lemma('horizontal.a.01.horizontal').antonyms())


print(dir(wn.synset('harmony.n.02')))


from nltk.corpus import wordnet as wn
right = wn.synset('right_whale.n.01')
orca = wn.synset('orca.n.01')
minke = wn.synset('minke_whale.n.01')
tortoise = wn.synset('tortoise.n.01')
novel = wn.synset('novel.n.01')
print(right.lowest_common_hypernyms(minke))
print(right.lowest_common_hypernyms(orca))
print(right.lowest_common_hypernyms(tortoise))
print(right.lowest_common_hypernyms(novel))


print(wn.synset('baleen_whale.n.01').min_depth())
print(wn.synset('whale.n.02').min_depth())
print(wn.synset('vertebrate.n.01').min_depth())
print(wn.synset('entity.n.01').min_depth())


print(right.path_similarity(minke))
print(right.path_similarity(orca))
print(right.path_similarity(tortoise))
print(right.path_similarity(novel))
