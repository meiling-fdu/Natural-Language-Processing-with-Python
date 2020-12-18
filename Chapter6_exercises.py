import nltk
from nltk.corpus import names, senseval, movie_reviews, nps_chat, brown, ppattach
import random

# 2
# Gender Identification

names = ([(name, 'male') for name in names.words('male.txt')] +
         [(name, 'female') for name in names.words('female.txt')])
random.shuffle(names)
print(len(names))


def gender_features(word):
    features = {
            'suffix1': word[-1:],
            'suffix2': word[-2:],
            'first_letter': word[0].lower(),
            'length': len(word),
            }
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = word.lower().count(letter)
        features["has(%s)" % letter] = (letter in word.lower())
    return features


train_names = names[1000:]
devtest_names = names[500:1000]
test_names = names[:500]

train_set = [(gender_features(n), g) for (n, g) in train_names]
devtest_set = [(gender_features(n), g) for (n, g) in devtest_names]
test_set = [(gender_features(n), g) for (n, g) in test_names]

classifier = nltk.DecisionTreeClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))
print(nltk.classify.accuracy(classifier, test_set))
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))
print(nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(5))


# 3
# Exploiting Context
instances = senseval.instances('hard.pos')

def pos_features(inst):
    p = inst.position
    features = {"left_tag": inst.context[p - 1][1],
                "right_tag": inst.context[p + 1][1],
                }
    return features

featuresets = [(pos_features(inst), inst.senses) for inst in instances]
print(len(featuresets))
print(featuresets[:5])
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier = nltk.DecisionTreeClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))


# 4
# Document Classification
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = [key for key, _ in all_words.most_common(2000)]
# word_features = list(all_words)[:2000] # another method


def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(30))


# 5
# Identifying Dialogue Act Types
posts = nps_chat.xml_posts()[:10000]


def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains(%s)' % word.lower()] = True
    return features


featuresets = [(dialogue_act_features(post.text), post.get('class'))
               for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier = nltk.DecisionTreeClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier = nltk.MaxentClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))


# 6
# Exploiting Context
def pos_features(tagged_sent, i):
    features = {}
    features["prev-word"] = tagged_sent[i-1][0]
    features["following-word"] = tagged_sent[i+1][0]
    features["prev-tag"] = tagged_sent[i-1][1]
    features["following-tag"] = tagged_sent[i+1][1]
    return features


tagged_sents = brown.tagged_sents()
featuresets = []
for tagged_sent in tagged_sents:
    for i, (word, tag) in enumerate(tagged_sent):
        if word in ['strong']:
            # print(tagged_sent)
            featuresets.append((pos_features(tagged_sent, i), tag))
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(5))

tagged_sents = brown.tagged_sents()
featuresets = []
for tagged_sent in tagged_sents:
    for i, (word, tag) in enumerate(tagged_sent):
        if word in ['powerful']:
            featuresets.append((pos_features(tagged_sent, i), tag))
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(5))


# 7
# Sequence Classification
def dialogue_act_features(post, i):
    features = {}
    text = nltk.word_tokenize(post.text)
    for word in text:
        features['contains(%s)' % word.lower()] = True
    if i > 0:
        features["prev-tag"] = posts[i-1].get('class')
    else:
        features["prev-tag"] = '<Start>'
    return features

posts = nltk.corpus.nps_chat.xml_posts()[:700]
# for (i, post) in enumerate(posts[708:10000]):
#     print(i, post.text, post.get('class'))
# print(posts[710].get('class'))

featuresets = []
for (i, post) in enumerate(posts):
    print(i, post.text, post.get('class'))
    featuresets.append((dialogue_act_features(post, i), post.get('class')))

print(len(featuresets))
size = int(len(featuresets) * 0.1)
print(size)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))


# 9 Problem X
def inst_features(n1, n2):
    features = {
        'noun_1': n1,
        'noun_2': n2,
    }
    return features
nattach = [(inst.noun1, inst.prep, inst.noun2)
           for inst in ppattach.attachments('training')
           if inst.attachment == 'N']
train_set = [(inst_features(inst[0], inst[2]), inst[1]) for inst in nattach]
print(train_set)
classifier = nltk.NaiveBayesClassifier.train(train_set)
inst = [('team', 'researchers')]
print(nltk.classify.accuracy(classifier, inst))
