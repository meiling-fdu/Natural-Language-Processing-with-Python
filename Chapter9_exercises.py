import nltk

# 1
grammar = nltk.data.load('9_1_1.fcfg')
print(grammar)
tokens_1 = "I am happy".split()
tokens_2 = "she is happy".split()
tokens_3 = "she am happy".split()

parser = nltk.load_parser('9_1_1.fcfg')
for tree in parser.parse(tokens_1):
    print(tree)
for tree in parser.parse(tokens_2):
    print(tree)

parser = nltk.load_parser('9_1_2.fcfg')
for tree in parser.parse(tokens_1):
    print(tree)
for tree in parser.parse(tokens_2):
    print(tree)


# 2
tokens_1_1 = "The boy sings".split()
tokens_1_2 = "Boy sings".split()

tokens_2_1 = "The boys sing".split()
tokens_2_2 = "Boys sing".split()

tokens_3_1 = "The water is precious".split()
tokens_3_2 = "Water is precious".split()

parser = nltk.load_parser('9_2.fcfg')
for tree in parser.parse(tokens_1_1):
    print(tree)
for tree in parser.parse(tokens_1_2):
    print(tree)
for tree in parser.parse(tokens_2_1):
    print(tree)
for tree in parser.parse(tokens_2_2):
    print(tree)
for tree in parser.parse(tokens_3_1):
    print(tree)
for tree in parser.parse(tokens_3_2):
    print(tree)


# 3
def subsumes(fs1, fs2):
    if fs1.unify(fs2) == fs2:
        return True


fs1 = nltk.FeatStruct(NUMBER=74)
fs2 = nltk.FeatStruct(NUMBER=74, STREET='rue Pascal')

print(subsumes(fs1, fs2))


# 4
tokens = "the student from France with good grades walks".split()
parser = nltk.load_parser('9_4.fcfg')
for tree in parser.parse(tokens):
    print(tree)


# 5 (incomplete)
cp = nltk.load_parser('grammars/book_grammars/german.fcfg', trace=2)
tokens = 'ich folge den Katze'.split()
for tree in cp.parse(tokens):
    print(tree)


# 7
cp = nltk.load_parser('grammars/book_grammars/german.fcfg')
tokens = 'ich folge den Katze'.split()
tag = 0
for tree in cp.parse(tokens):
    if tree:
        tag = 1
        print(tree)
if tag == 0:
    print('FAIL')


# 8
fs1 = nltk.FeatStruct("[A = ?x, B= [C = ?x]]")
fs2 = nltk.FeatStruct("[B = [D = d]]")
fs3 = nltk.FeatStruct("[B = [C = d]]")
fs4 = nltk.FeatStruct("[A = (1)[B = b], C->(1)]")
fs5 = nltk.FeatStruct("[A = (1)[D = ?x], C = [E -> (1), F = ?x] ]")
fs6 = nltk.FeatStruct("[A = [D = d]]")
fs7 = nltk.FeatStruct("[A = [D = d], C = [F = [D = d]]]")
fs8 = nltk.FeatStruct("[A = (1)[D = ?x, G = ?x], C = [B = ?x, E -> (1)] ]")
fs9 = nltk.FeatStruct("[A = [B = b], C = [E = [G = e]]]")
fs10 = nltk.FeatStruct("[A = (1)[B = b], C -> (1)]")

print(fs2.unify(fs1))
print(fs1.unify(fs3))
print(fs4.unify(fs5))
print(fs5.unify(fs6))
print(fs5.unify(fs7))
print(fs5.unify(fs6))
print(fs8.unify(fs9))
print(fs8.unify(fs10))


# 9
fs1 = nltk.FeatStruct("[A = ?x, B= [C = ?x]]")
fs2 = nltk.FeatStruct("[ADDRESS1=?x, ADDRESS2=?x]")
print(fs1)
print(fs2)


# 12
parser = nltk.load_parser('9_12.fcfg')
token_1 = "The farmer loaded sand into the cart".split()
token_2 = "The farmer loaded the cart with sand".split()
for tree in parser.parse(token_1):
    print(tree)
for tree in parser.parse(token_2):
    print(tree)
