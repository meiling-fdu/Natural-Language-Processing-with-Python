from __future__ import division
import nltk
from nltk.book import text1, text2, text5, text6, sent1, sent3

print(text2.common_contexts(['monstrous', 'very']))

print(text2.similar('monstrous'))

print(text2.similar('very'))

print(text2.dispersion_plot(['very']))

text2.generate()

print(len(set(text1)))

# count occurrence of each word on average
print(len(text2) / len(set(text2)))

print(text5.count('lol') / len(text5))

print(sorted(sent1))

lml = ['Hello', 'mei', 'ling', 'li', '!']
print(lml)
lml[1:3] = ['yibo']
print(lml)

test = '*'.join(['yibo', 'and', 'zz'])
print(test)

print(test.split('*'))

freq1 = nltk.FreqDist(text1)
vocabulary = freq1.keys()
print(freq1['whale'])
print(freq1)
print(freq1.values(), vocabulary, freq1['am'])
for item, value in enumerate(freq1):
    print(item, value)
freq1.plot(50, cumulative=True)
print(len(freq1.hapaxes()))

V = set(text1)
long_words = [w for w in V if len(w) > 15]
print(long_words[:15])

freq5 = nltk.FreqDist(text5)
set5 = set(text5)
long_words = sorted([w for w in set5 if len(w) > 7 and freq5[w] > 7])
print(long_words[:20])

print(text1.collocations())

fdist1 = nltk.FreqDist([len(w) for w in text1])
print(fdist1.N())
print(fdist1.items())
print(fdist1[3], fdist1.max(), fdist1.freq(3))

print(text1)
print([w for w in sent1 if w.startswith('s')])
print([w for w in sent1 if 's' not in w])
print([w for w in set(text1) if not w.islower()])

from nltk.misc import babelize_shell

babelize_shell()

nltk.chat.chatbots()

text2.collocations()

print(sent3, sent3.index('the'))

print(len(sorted(set([w.lower() for w in text1]))))
print(len(sorted([w.lower() for w in set(text1)])))

# 21
print(text2[-2:])

# 22
four_letter_words = [w for w in text5 if len(w) == 4]
print(four_letter_words)
freq5 = nltk.FreqDist(four_letter_words)
print(freq5.items())
print(freq5.keys())
print(freq5.most_common())

# 23
for w in text6:
    if w.isupper():
        print(w)

# 24
lst = [w for w in text6 if w.endswith('ize')]
print(lst)
lst = [w for w in text6 if 'pt' in w]
print(lst)
lst = [w for w in text6 if w.istitle()]
print(lst)

# 26
print(sum([len(w) for w in text1]))

# 29
print(set(sent3) < set(sent1))
print(set(text5) < set(text1))
