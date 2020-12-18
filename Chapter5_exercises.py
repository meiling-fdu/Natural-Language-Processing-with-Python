import pprint
import nltk
import pylab
from nltk.corpus import brown, gutenberg



# 1
lst1 = 'British Left Waffles on Falkland Islands'
lst2 = 'Juvenile Court to Try Shooting Defendant'
text1 = nltk.word_tokenize(lst1)
print(nltk.pos_tag(text1))
text2 = nltk.word_tokenize(lst2)
print(nltk.pos_tag(text2))


# 2
brown_news_tagged = brown.tagged_words(tagset='universal')
w = 'getting'
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged if word == w)
print(tag_fd.most_common())
tag_fd.plot(cumulative=False)


# 3
tokens = nltk.word_tokenize('They wind back the clock, while we chase after the wind.')
print(nltk.pos_tag(tokens))


# 5
d = dict()
print(d)
d['letter'] = 'romance'
print(d['xyz'])


# 6
d = {'abc': 'novel'}
print(d)
del d['abc']
print(d)


# 7 this can be used for adding or changing the content of some words
d1 = {'abc': 'novel'}
d2 = {'abc': 'original', 'new': 'movie'}
d1.update(d2)
print(d1)


# 8
e = {}
e['headword'] = 'leave'
e['part-of-speech'] = ['VB', 'NN']
e['sense'] = 'go away'
e['example'] = 'She will leave soon.'
print(e)


# 9
brown_news_tagged = brown.tagged_words(tagset='universal')
tag_fd_go = nltk.FreqDist(tag for (word, tag) in brown_news_tagged if word == 'go')
print(tag_fd_go.most_common())
tag_fd_went = nltk.FreqDist(tag for (word, tag) in brown_news_tagged if word == 'went')
print(tag_fd_went.most_common())

brown_words = nltk.Text(brown.words(categories='news'))
print(brown_words.concordance('go'))
print(brown_words.concordance('went'))



# 10-11-12
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]

unigram_tagger = nltk.UnigramTagger(train_sents)
print(unigram_tagger.tag(brown_sents[4300]))
print(unigram_tagger.evaluate(test_sents))

# help(nltk.AffixTagger)
affix_tagger = nltk.AffixTagger(train_sents, affix_length=-1, min_stem_length=2)
print(affix_tagger.tag(brown_sents[4300]))
print(affix_tagger.evaluate(test_sents))

bigram_tagger = nltk.BigramTagger(train_sents)
print(bigram_tagger.tag(brown_sents[2007]))
print(bigram_tagger.tag(brown_sents[4300]))
print(bigram_tagger.evaluate(train_sents))
print(bigram_tagger.evaluate(test_sents))


# 13
d = {'year': 2020, 'month': 11, 'day': 12}
print('%s-%s-%s' % (d['year'], d['month'], d['day']))
print('{year}-{month}-{day}'.format(year=d['year'], month = d['month'], day = d['day']))
print(f"{d['year']}-{d['month']}-{d['day']}")


# 14
brown_tagged_words = brown.tagged_words()  # tagset='universal'
tag_list = sorted(set([tag for _, tag in brown_tagged_words]))
print(tag_list)


# 15
brown_tagged_words = brown.tagged_words()
cfd = nltk.ConditionalFreqDist(brown_tagged_words)
# (1)
lst = [w for w in cfd.conditions() if 'NN' in cfd[w] and 'NNS' in cfd[w+'s']]
print(len(lst), lst)
for w in lst:
    if not cfd[w+'s']['NNS'] > cfd[w]['NN']:
        lst.remove(w)
print(len(lst), lst)
# (2)-method 1
max = 0
for w in cfd.conditions():
    if len(cfd[w]) > max:
        max = len(cfd[w])
lst = [w for w in cfd.conditions() if len(cfd[w]) == max]
print(max, lst)
for w in lst:
    print(cfd[w].most_common())
# (2)-method 2
tag_word_list = [(w, len(cfd[w])) for w in cfd.conditions()]
tags_rank = sorted(tag_word_list, key=lambda x: x[1], reverse=True)
print(tags_rank[:20])
# (3)
tag_list = [tag for _, tag in brown_tagged_words]
fdist = nltk.FreqDist(tag_list)
tag_rank = sorted(fdist.keys(), key=lambda x: fdist[x], reverse=True)
print(tag_rank[:20])
# or directly:
print(fdist.most_common(20))


# 16

def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))


def display():
    import pylab
    words_by_freq = list(nltk.FreqDist(brown.words(categories='news')))
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(16)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()


display()


# 17
brown_tagged_sents = brown.tagged_sents(categories='news')
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
likely_tags = dict((word, cfd[word].max()) for word in fd.keys())
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
print(baseline_tagger.evaluate(brown_tagged_sents))  # 0.9349006503968017

maxs_num, total_num = 0, 0
for w, tags in cfd.items():
    maxs_num += tags[tags.max()]
    total_num += tags.N()
rst = maxs_num / total_num
print(rst)  # 0.9349006503968017
# ref: https://github.com/walshbr/nltk/blob/master/ch_five/17.py
avgs = []
for w, tags in cfd.items():
    n = tags.N()
    max_num = tags[tags.max()]
    avg = max_num / n
    avgs.append(avg)
rst = sum(avgs) / len(avgs)
print(rst)  # 0.9644662324868203


# 18
tokens = brown.words(categories='news')
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
same_tag_wordlist = [w for w in cfd.conditions() if len(cfd[w]) == 1]
print(len(same_tag_wordlist)/len(cfd.conditions()))
ambi_tag_wordlist = [w for w in cfd.conditions() if len(cfd[w]) > 1]
print(len(ambi_tag_wordlist), len(cfd.conditions())-len(same_tag_wordlist))
ambi_tokens = [w for w in tokens if w in ambi_tag_wordlist]
print(len(ambi_tokens)/len(tokens))


# 19
print(nltk.tag.api.__file__)


# 20
brown_tagged_words = brown.tagged_words()
cfd = nltk.ConditionalFreqDist(brown.tagged_words())

MD_wordlist = [w for w, tag in brown_tagged_words if tag == 'MD']
MD_wordlist = [w for w in cfd.conditions() if cfd[w]['MD']]
print(sorted(set(MD_wordlist)))

biwordlist = [w for w in cfd.conditions() if cfd[w]['NNS'] and cfd[w]['VBZ']]
print(biwordlist)

phrase_list = []
for (w1, t1), (w2, t2), (w3, t3) in list(nltk.trigrams(brown.tagged_words())):
    if t1 == 'IN' and t2 == 'DT' and t3 == 'NN':
        phrase_list.append((w1, w2, w3))
print(phrase_list)

masculine_prop = ['he', 'him', 'his', 'himself']
feminine_prop = ['she', 'her', 'hers', 'herself']
men_list = [w for w in brown.words() if w.lower() in masculine_prop]
women_list = [w for w in brown.words() if w.lower() in feminine_prop]
print(len(men_list)/len(women_list))


# 21 ref: https://github.com/walshbr/nltk/blob/master/ch_five/21.py
brown_tagged_words = brown.tagged_words(tagset='universal')
bigrams = list(nltk.bigrams(brown_tagged_words))
print(bigrams[:10])
verbs = ['love', 'like', 'adore', 'prefer']
qualifiers = set([w1[0] for w1, w2 in bigrams if w2[0] in verbs and w2[1] == 'VERB' and w1[1] == 'ADV'])
print(qualifiers)

tagged_text = brown.tagged_words(tagset='universal')
tagged_bigrams = list(nltk.bigrams(tagged_text))

for bigram in tagged_bigrams:
    zipped_tag = [list(t) for t in zip(*bigram)]
    print(zipped_tag)
    if zipped_tag[0][1] in ['adore', 'love', 'prefer', 'like'] and zipped_tag[1][1] == 'VERB' and zipped_tag[1][
        0] == 'ADV':
        print(zipped_tag[0][0])


# 22-23
patterns = [
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*ed$', 'VBD'),  # simple past
    (r'.*en$', 'VBN'),  # past participle (added)
    (r'.*es$', 'VBZ'),  # 3rd singular present
    (r'.*ould$', 'MD'),  # modals
    (r'.*\'s$', 'NN$'),  # possessive nouns
    (r'.*s$', 'NNS'),  # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')  # nouns (default)
]
brown_tagged_sents = brown.tagged_sents()
size = int(len(brown_tagged_sents) * 0.9)
train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]
regexp_tagger = nltk.RegexpTagger(patterns)
print(regexp_tagger.evaluate(test_sents))


# 24
brown_tagged_sents = brown.tagged_sents()
size = int(len(brown_tagged_sents) * 0.9)
train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]

cfd = nltk.ConditionalFreqDist(
    (n, nltk.NgramTagger(n, train=train_sents).evaluate(test_sents))
    for n in range(1, 7)
)
cfd.tabulate()


# 25
sinica_tagged_sents = nltk.corpus.sinica_treebank.tagged_sents()
size = int(len(sinica_tagged_sents) * 0.9)
print(size)
train_sents, test_sents = sinica_tagged_sents[:size], sinica_tagged_sents[size:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
print(t1.evaluate(test_sents))
print(t1.tag(test_sents[823]))

brown_tagged_sents = brown.tagged_sents()
size = int(len(brown_tagged_sents) * 0.9)
train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
print(t1.evaluate(test_sents))


# 26
def performance(train_sents, test_sents):
    unigram_tagger = nltk.UnigramTagger(train=train_sents, backoff=nltk.DefaultTagger('NN'))
    return unigram_tagger.evaluate(test_sents)
def display():
    import pylab
    brown_tagged_sents = brown.tagged_sents()  # categories='news'
    print(len(brown_tagged_sents))
    sizes = 2 ** pylab.arange(16)  # 13
    perfs = []
    for size in sizes:
        train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]
        perfs.append(performance(train_sents, test_sents))
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Unigram Tagger Performance with Varying Training Data Size')
    pylab.xlabel('Training Data Size')
    pylab.ylabel('Performance')
    pylab.show()
display()


# 27-28

brown_tagged_sents = brown.tagged_sents(categories='editorial')
size = int(len(brown_tagged_sents) * 0.9)
train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, cutoff=2, backoff=t1)

test_tags = [tag for sent in brown.sents(categories='editorial')
             for (word, tag) in t2.tag(sent)]
gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')]
print(nltk.ConfusionMatrix(gold_tags, test_tags))
print(t2.evaluate(test_sents))

# simplified data

def collapse(tagged_sents, collapse_size):
    conversion_dict = {}
    tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')]
    # print(tags)
    for tag in tags:
        if len(tag) > 2:
            conversion_dict[tag] = tag[:collapse_size]
        else:
            conversion_dict[tag] = tag
    # print(conversion_dict)
    new_tagged_sents = []
    for sent in tagged_sents:
        new_sent = []
        for w, tag in sent:
            new_sent.append((w, conversion_dict[tag]))
        new_tagged_sents.append(new_sent)
    return new_tagged_sents


brown_tagged_sents = collapse(brown.tagged_sents(categories='editorial'), 2)
print(brown_tagged_sents[0])
size = int(len(brown_tagged_sents) * 0.9)
train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, cutoff=2, backoff=t1)
print(t2.evaluate(test_sents))

brown_tagged_sents = collapse(brown.tagged_sents(categories='editorial'), 1)
print(brown_tagged_sents[0])
size = int(len(brown_tagged_sents) * 0.9)
train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, cutoff=2, backoff=t1)
print(t2.evaluate(test_sents))


# 29

def tag_None(tagged_sent):
    for w, tag in tagged_sent:
        if tag is None:
            return True
    return False
brown_tagged_sents = brown.tagged_sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]
brown_sents = brown.sents(categories='news')
tr_sents, te_sents = brown_sents[:size], brown_sents[size:]
print(len(brown_tagged_sents), len(brown_sents))
bigram_tagger = nltk.BigramTagger(train_sents)
train_fail_sents = [bigram_tagger.tag(sent) for sent in tr_sents if tag_None(bigram_tagger.tag(sent))]
print(train_fail_sents[:2])
test_fail_sents = [bigram_tagger.tag(sent) for sent in te_sents if tag_None(bigram_tagger.tag(sent))]
print(test_fail_sents[:2])
print(bigram_tagger.tag(brown_sents[2007]))
print(bigram_tagger.tag(brown_sents[4203]))
print(bigram_tagger.evaluate(test_sents))


# 30
def process(tagged_sents):
    fdist = nltk.FreqDist(brown.words())
    common_wordlist = [w for w, _ in fdist.most_common(500)]
    new_tagged_sents = []
    for sent in tagged_sents:
        new_sent = []
        for w, tag in sent:
            if w not in common_wordlist:
                w = 'UNK'
            new_sent.append((w, tag))
        new_tagged_sents.append(new_sent)
    return new_tagged_sents


brown_tagged_sents = brown.tagged_sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
bigram_tagger = nltk.BigramTagger(train=train_sents, backoff=t1)
print(t0.evaluate(gold=test_sents), t1.evaluate(gold=test_sents), bigram_tagger.evaluate(gold=test_sents))


processed_tagged_sents = process(brown_tagged_sents)
size = int(len(processed_tagged_sents) * 0.9)
train_sents, test_sents = processed_tagged_sents[:size], processed_tagged_sents[size:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
bigram_tagger = nltk.BigramTagger(train=train_sents, backoff=t1)
print(t0.evaluate(gold=test_sents), t1.evaluate(gold=test_sents), bigram_tagger.evaluate(gold=test_sents))


# 31
def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))
def display():
    import pylab
    words_by_freq = list(nltk.FreqDist(brown.words(categories='news')))
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.semilogx(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()
display()


# 32 ref: https://github.com/walshbr/nltk/blob/master/ch_five/32.py
brown_tagged_sents = brown.tagged_sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]

trainer = nltk.brill_trainer.BrillTaggerTrainer(
    initial_tagger=nltk.DefaultTagger('NN'),
    templates=nltk.tag.brill.nltkdemo18(),
    trace=3,
    deterministic=True
)

tagger = trainer.train(train_sents, max_rules=20, min_score=2, min_acc=None)
print(tagger.evaluate(test_sents))


# 33 ref: https://github.com/walshbr/nltk/blob/master/ch_five/33.py
tagged_words = set(brown.tagged_words(categories='news')[:3000])
# print(tagged_words)
bigrams = list(nltk.bigrams(tagged_words))
# print(bigrams)

def singleDict(given_word, given_tag):
    follow_tags = set()
    for bigram in bigrams:
        if bigram[0] == (given_word, given_tag):
            follow_tags.add(bigram[1][1])
    return follow_tags

def wholeDict(tagged_words):
    dict = {}
    for (word, tag) in tagged_words:
        dict[(word, tag)] = singleDict(word, tag)
    return dict

dict = wholeDict(tagged_words)
print(dict[('go', 'VB')])


# 34
brown_tagged_words = brown.tagged_words()
cfd = nltk.ConditionalFreqDist(brown_tagged_words)


def calc_num_list():
    num_list = [0 for x in range(1, 16)]
    # print(num_list)
    for w in cfd.conditions():
        # print(cfd[w], len(cfd[w]))
        num_list[len(cfd[w])] += 1
    return num_list


def print_table(num_list):
    print('%s\t%s' % ('tag_num', 'word_num'))
    for i in range(1, 14):
        print('%d\t\t%d' % (i, num_list[i]))


def examples(w, tagset):
    # word = 'that'
    # tagset = {'DT-NC', 'CS-HL', 'CS-NC', 'WPS', 'NIL', 'WPS-NC', 'DT', 'WPS-HL', 'WPO', 'CS', 'WPO-NC', 'QL'}
    brown_tagged_sents = brown.tagged_sents()
    num = len(tagset)
    flag = nltk.defaultdict(int)
    for sent in brown_tagged_sents:
        for tag in tagset:
            if (w, tag) in sent and flag[tag] == 0:
                print(sent)
                flag[tag] = 1


def max_tag_sents(num_list):
    for i in range(len(num_list)):
        if num_list[i]:
            max_tag_num = i
    print(max_tag_num)
    for w in cfd.conditions():
        if len(cfd[w]) == max_tag_num:
            print(w)
            tagset = set([tag for word, tag in brown_tagged_words if word == w])
            print(tagset)
            examples(w, tagset)


nums_list = calc_num_list()
print_table(nums_list)
max_tag_sents(nums_list)

# 35 ref: https://github.com/walshbr/nltk/blob/master/ch_five/35.py
brown_tagged_words = brown.tagged_words()
cfd = nltk.ConditionalFreqDist(brown_tagged_words)
bigrams = list(nltk.bigrams(brown_tagged_words))
words_following_must = []
for (p1, p2) in bigrams:
    # print(p1)
    if p1[0] == 'must':
        words_following_must.append((p1, p2))
print(words_following_must[:3])
tags_following_must = set([tag for (_, _), (_, tag) in words_following_must])
print(tags_following_must)

for (p1, p2) in words_following_must:
    print(p1[0] + " " + p1[1] + " " + p2[0] + " " + p2[1])
    if p2[1] in ['BE', 'BE-HL', 'NN', 'NNS', 'NP-HL']:
        print("Context is likely epistemic")
    elif p2[1] in ['HV', 'HV-TL', 'DO', 'RB', 'RB-HL', 'VB', 'VB-HL', 'VB-TL', 'VBZ']:
        print("Context is likely deontic")
    else:
        print("Context is unclear")


# 36

patterns = [
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*ed$', 'VBD'),  # simple past
    (r'.*en$', 'VBN'),  # past participle (added)
    (r'.*es$', 'VBZ'),  # 3rd singular present
    (r'.*ould$', 'MD'),  # modals
    (r'.*\'s$', 'NN$'),  # possessive nouns
    (r'.*s$', 'NNS'),  # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')  # nouns (default)
]

brown_tagged_sents = brown.tagged_sents()
size = int(len(brown_tagged_sents) * 0.9)
train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]

# combination
t = nltk.defaultdict()
t[0] = nltk.RegexpTagger(patterns)

for i in range(1, 6):
    t[i] = nltk.NgramTagger(i, train=train_sents, backoff=t[i-1])
    # print(t[i].evaluate(test_sents))

# model size test
def performance(s):
    brown_tagged_sents = brown.tagged_sents()[:s]
    size = int(len(brown_tagged_sents) * 0.9)
    train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]

    t0 = nltk.RegexpTagger(patterns)
    t1 = nltk.NgramTagger(1, train=train_sents, backoff=t0)
    t2 = nltk.NgramTagger(2, train=train_sents, backoff=t1)
    tagger = nltk.NgramTagger(3, train=train_sents, backoff=t2)
    return tagger.evaluate(test_sents)

def display():
    sizes = 2 ** pylab.arange(10, 16)
    perfs = [performance(size) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Combined Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()
display()


# 37 ref: https://github.com/walshbr/nltk/blob/master/ch_five/37.py
tagged_sents = brown.tagged_sents(categories='news')

size = int(len(tagged_sents) * 0.9)
train_sents = tagged_sents[:size]
test_sents = tagged_sents[size:]


class PreviousTagger(nltk.UnigramTagger):
    json_tag = 'nltk.tag.sequential.PreviousTagger'

    def context(self, tokens, index, history):
        if index == 0:
            return None
        else:
            return history[index-1]


t0 = nltk.DefaultTagger('NN')
t1 = PreviousTagger(train_sents, backoff=t0)
t2 = nltk.UnigramTagger(train_sents, backoff=t1)
t3 = nltk.BigramTagger(train_sents, backoff=t2)
t4 = nltk.TrigramTagger(train_sents, backoff=t3)

pprint.pprint(t4.tag(['I', 'like', 'to', 'blog', 'on', 'Kim\'s', 'blog']))


# 42
def performance(s, split_way):
    if split_way == 1:
        brown_tagged_sents = brown.tagged_sents()[:s]
        size = int(len(brown_tagged_sents) * 0.9)
        train_sents, test_sents = brown_tagged_sents[:size], brown_tagged_sents[size:]
    elif split_way == 2:
        train_sents = brown.tagged_sents(categories='editorial')[:s]
        test_sents = brown.tagged_sents(categories='news')[:s]
    else:
        train_sents = brown.tagged_sents('cf33')[:s]
        test_sents = brown.tagged_sents('cg33')[:s]

    t0 = nltk.DefaultTagger('NN')
    t1 = nltk.NgramTagger(1, train=train_sents, backoff=t0)
    t2 = nltk.NgramTagger(2, train=train_sents, backoff=t1)
    tagger = nltk.NgramTagger(3, train=train_sents, backoff=t2)
    return tagger.evaluate(test_sents)


def display():
    split_way = {'sentence': 1, 'genre': 2, 'source': 3}
    for key, value in split_way.items():
        sizes = 2 ** pylab.arange(5, 16)
        perfs = [performance(size, value) for size in sizes]
        pylab.plot(sizes, perfs, '-o', label=key)
        pylab.title('Tagger Performance with Varying Split Ways')
        pylab.xlabel('Model Size')
        pylab.legend()
        pylab.ylabel('Performance')
    pylab.show()


display()
