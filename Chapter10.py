import nltk
from nltk import load_parser
from nltk.sem import chat80

# 10.1 Querying a Database
nltk.data.show_cfg('grammars/book_grammars/sql0.fcfg')
cp = load_parser('grammars/book_grammars/sql0.fcfg', trace=3)
query = 'What cities are located in China'
trees = list(cp.parse(query.split()))
answer = trees[0].label()['SEM']
answer = [s for s in answer if s]
q = ' '.join(answer)
print(q)

rows = chat80.sql_query('corpora/city_database/city.db', q)
for r in rows: print(r[0], end=" ")


nltk.data.show_cfg('grammars/book_grammars/sql1.fcfg')
cp = load_parser('grammars/book_grammars/sql1.fcfg')
query = 'What cities are in China and have populations above 1,000,000'
trees = list(cp.parse(query.split()))
answer = trees[0].label()['SEM']
answer = [s for s in answer if s]
q = ' '.join(answer)
print(q)


# 10.2 Propositional Logic
nltk.boolean_ops()

read_expr = nltk.sem.Expression.fromstring
print(read_expr('-(P & Q)'))
print(read_expr('P & Q'))
print(read_expr('P | (R -> Q)'))
print(read_expr('P <-> -- P'))


read_expr = nltk.sem.Expression.fromstring
lp = nltk.sem.Expression.fromstring
SnF = read_expr('SnF')
NotFnS = read_expr('-FnS')
R = read_expr('SnF -> -FnS')
prover = nltk.Prover9()
print(prover.prove(NotFnS, [SnF, R]))


val = nltk.Valuation([('P', True), ('Q', True), ('R', False)])
print(val['P'])
dom = set()
g = nltk.Assignment(dom)
m = nltk.Model(dom, val)
print(m.evaluate('(P & Q)', g))
print(m.evaluate('-(P & Q)', g))
print(m.evaluate('(P & R)', g))
print(m.evaluate('(P | R)', g))


# 10.3 First-Order Logic
read_expr = nltk.sem.Expression.fromstring
expr = read_expr('walk(angus)', type_check=True)
print(expr.argument)
print(expr.argument.type)
print(expr.function)
print(expr.function.type)

sig = {'walk': '<e, t>'}
expr = read_expr('walk(angus)', type_check=True, signature=sig)
print(expr.function.type)

read_expr = nltk.sem.Expression.fromstring
print(read_expr('dog(cyril)').free())
print(read_expr('dog(x)').free())
print(read_expr('own(angus, cyril)').free())
print(read_expr('exists x.dog(x)').free())
print(read_expr('((some x. walk(x)) -> sing(x))').free())
print(read_expr('exists x.own(y, x)').free())


read_expr = nltk.sem.Expression.fromstring
lp = nltk.sem.Expression.fromstring
NotFnS = read_expr('-north_of(f, s)')
SnF = read_expr('north_of(s, f)')
R = read_expr('all x. all y. (north_of(x, y) -> -north_of(y, x))')
prover = nltk.Prover9()
print(prover.prove(NotFnS, [SnF, R]))

FnS = read_expr('north_of(f, s)')
print(prover.prove(FnS, [SnF, R]))


dom = {'b', 'o', 'c'}
v = """
bertie => b
olive => o
cyril => c
boy => {b}
girl => {o}
dog => {c}
walk => {o, c}
see => {(b, o), (c, b), (o, c)}
"""
val = nltk.Valuation.fromstring(v)
print(val)
print(val['walk'])

print(('o', 'c') in val['see'])
print(('b',) in val['boy'])

g = nltk.Assignment(dom, [('x', 'o'), ('y', 'c')])
print(g)

m = nltk.Model(dom, val)
print(m.evaluate('see(olive, y)', g))
print(m.evaluate('see(y, x)', g))

g.purge()
print(g)

print(m.evaluate('see(olive, y)', g))
print(m.evaluate('see(bertie, olive) & boy(bertie) & -walk(bertie)', g))

print(m.evaluate('exists x.(girl(x) & walk(x))', g))
print(m.evaluate('girl(x) & walk(x)', g.add('x', 'o')))

read_expr = nltk.sem.Expression.fromstring
fmla1 = read_expr('girl(x) | boy(x)')
print(m.satisfiers(fmla1, 'x', g))
fmla2 = read_expr('girl(x) -> walk(x)')
print(m.satisfiers(fmla2, 'x', g))
fmla3 = read_expr('walk(x) -> girl(x)')
print(m.satisfiers(fmla3, 'x', g))
print(m.evaluate('all x.(girl(x) -> walk(x))', g))
print(m.evaluate('exists x.(boy(x) -> walk(x))', g))


v2 = """
bruce => b
elspeth => e
julia => j
matthew => m
person => {b, e, j, m}
admire => {(j, b), (b, b), (m, e), (e, m)}
"""
val2 = nltk.Valuation.fromstring(v2)
dom2 = val2.domain
m2 = nltk.Model(dom2, val2)
g2 = nltk.Assignment(dom2)

fmla4 = read_expr('(person(x) -> exists y.(person(y) & admire(x, y)))')
print(m2.satisfiers(fmla4, 'x', g2))

fmla5 = read_expr('(person(y) & all x.(person(x) -> admire(x, y)))')
print(m2.satisfiers(fmla5, 'y', g2))

fmla6 = read_expr('(person(y) & all x.((x = bruce | x = julia) -> admire(x, y)))')
print(m2.satisfiers(fmla6, 'y', g2))

# My turn -1
v2_1 = """
bruce => b
elspeth => e
julia => j
matthew => m
person => {b, e, j, m}
admire => {(j, b), (b, b), (m, e), (e, m)}
"""
read_expr = nltk.sem.Expression.fromstring
val2_1 = nltk.Valuation.fromstring(v2_1)
dom2_1 = val2_1.domain
m2_1 = nltk.Model(dom2_1, val2_1)
g2_1 = nltk.Assignment(dom2_1)

print(m2_1.evaluate('all x. exist y.(admire(x, y))', g2_1))
print(m2_1.evaluate('exist y. all x.(admire(x, y))', g2_1))

# My turn -2
v2_2 = """
bruce => b
elspeth => e
julia => j
matthew => m
person => {b, e, j, m}
admire => {(j, b), (b, b), (m, b), (e, b)}
"""
read_expr = nltk.sem.Expression.fromstring
val2_2 = nltk.Valuation.fromstring(v2_2)
dom2_2 = val2_2.domain
m2_2 = nltk.Model(dom2_2, val2_2)
g2_2 = nltk.Assignment(dom2_2)

print(m2_2.evaluate('all x. exist y.(admire(x, y))', g2_2))
print(m2_2.evaluate('exist y. all x.(admire(x, y))', g2_2))


read_expr = nltk.sem.Expression.fromstring
a3 = read_expr('exists x.(man(x) & walks(x))')
c1 = read_expr('mortal(socrates)')
c2 = read_expr('-mortal(socrates)')
mb = nltk.Mace(5)
print(mb.build_model(None, [a3, c1]))
print(mb.build_model(None, [a3, c2]))
print(mb.build_model(None, [c1, c2]))

a4 = read_expr('exists y. (woman(y) & all x. (man(x) -> love(x,y)))')
a5 = read_expr('man(adam)')
a6 = read_expr('woman(eve)')
g = read_expr('love(adam,eve)')
mc = nltk.MaceCommand(g, assumptions=[a4, a5, a6])
print(mc.build_model())
print(mc.valuation)

a7 = read_expr('all x. (man(x) -> -woman(x))')
g = read_expr('love(adam,eve)')
mc = nltk.MaceCommand(g, assumptions=[a4, a5, a6, a7])
print(mc.build_model())
print(mc.valuation)


# 10.4 The Semantics of English Sentences
# The λ-Calculus
read_expr = nltk.sem.Expression.fromstring
expr = read_expr(r'\x.(walk(x) & chew_gum(x))')
print(expr)
print(expr.free())
print(read_expr(r'\x.(walk(x) & chew_gum(y))').free())

expr = read_expr(r'\x.(walk(x) & chew_gum(x))(gerald)')
print(expr)
print(expr.simplify())

print(read_expr(r'\x.\y.(dog(x) & own(y, x))(cyril)').simplify())
print(read_expr(r'\x y.(dog(x) & own(y, x))(cyril, angus)').simplify())

expr1 = read_expr('exists x.P(x)')
print(expr1)
expr2 = expr1.alpha_convert(nltk.sem.Variable('z'))
print(expr2)
print(expr1 == expr2)

expr3 = read_expr('\P.(exists x.P(x))(\y.see(y, x))')
print(expr3)
print(expr3.simplify())


# Transitive Verbs
read_expr = nltk.sem.Expression.fromstring
tvp = read_expr(r'\X x.X(\y.chase(x,y))')
np = read_expr(r'(\P.exists x.(dog(x) & P(x)))')
vp = nltk.sem.ApplicationExpression(tvp, np)
print(vp)
print(vp.simplify())

parser = load_parser('grammars/book_grammars/simple-sem.fcfg', trace=0)
sentence = 'Angus gives a bone to every dog'
tokens = sentence.split()
for tree in parser.parse(tokens):
    print(tree.label()['SEM'])

sents = ['Irene walks', 'Cyril bites an ankle']
grammar_file = 'grammars/book_grammars/simple-sem.fcfg'
for results in nltk.interpret_sents(sents, grammar_file):
    for (synrep, semrep) in results:
        print(synrep)

v = """
bertie => b
olive => o
cyril => c
boy => {b}
girl => {o}
dog => {c}
walk => {o, c}
see => {(b, o), (c, b), (o, c)}
"""
val = nltk.Valuation.fromstring(v)
g = nltk.Assignment(val.domain)
m = nltk.Model(val.domain, val)
sent = 'Cyril sees every boy'
grammar_file = 'grammars/book_grammars/simple-sem.fcfg'
results = nltk.evaluate_sents([sent], grammar_file, m, g)[0]
for (syntree, semrep, value) in results:
    print(semrep)
    print(value)

# 10.5 Quantifier Ambiguity Revisited
from nltk.sem import cooper_storage as cs
sentence = 'every girl chases a dog'
trees = cs.parse_with_bindops(sentence, grammar='grammars/book_grammars/storage.fcfg')
semrep = trees[0].label()['SEM']
cs_semrep = cs.CooperStore(semrep)
print(cs_semrep.core)
for bo in cs_semrep.store:
    print(bo)

cs_semrep.s_retrieve(trace=True)

for reading in cs_semrep.readings:
    print(reading)


# 10.5 Discourse Semantics
# Discourse Representation Theory
read_dexpr = nltk.sem.DrtExpression.fromstring
drs1 = read_dexpr('([x, y], [angus(x), dog(y), own(x, y)])')
print(drs1)
drs1.draw()
print(drs1.fol())

drs2 = read_dexpr('([x], [walk(x)]) + ([y], [run(y)])')
print(drs2)
print(drs2.simplify())

drs3 = read_dexpr('([], [(([x], [dog(x)]) -> ([y],[ankle(y), bite(x, y)]))])')
print(drs3.fol())

drs4 = read_dexpr('([x, y], [angus(x), dog(y), own(x, y)])')
drs5 = read_dexpr('([u, z], [PRO(u), irene(z), bite(u, z)])')
drs6 = drs4 + drs5
print(drs6.simplify())
print(drs6.simplify().resolve_anaphora())

parser = load_parser('grammars/book_grammars/drt.fcfg', logic_parser=nltk.sem.drt.DrtParser())
trees = list(parser.parse('Angus owns a dog'.split()))
print(trees[0].label()['SEM'].simplify())


# Discourse Processing
dt = nltk.DiscourseTester(['A student dances', 'Every student is a person'])
dt.readings()
dt.add_sentence('No person dances', consistchk=True)
dt.retract_sentence('No person dances', verbose=True)
dt.add_sentence('A person dances', informchk=True)

# (fail to execute)
from nltk.tag import RegexpTagger
tagger = RegexpTagger(
    [('^(chases|runs)$', 'VB'),
     ('^(a)$', 'ex_quant'),
     ('^(every)$', 'univ_quant'),
     ('^(dog|boy)$', 'NN'),
     ('^(He)$', 'PRP')
     ]
)
rc = nltk.DrtGlueReadingCommand(depparser=nltk.MaltParser(tagger=tagger.tag, parser_dirname=r'D:\Program Files (x86)\maltparser-1.9.2'))
dt = nltk.DiscourseTester(['Every dog chases a boy', 'He runs'], rc)   # [sent.split() for sent in ['Every dog chases a boy', 'He runs']]
dt.readings()
dt.readings(show_thread_readings=True)
dt.readings(show_thread_readings=True, filter=True)