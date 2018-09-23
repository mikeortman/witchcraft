import sys
import json
import spacy
from multiprocessing import Pool, Lock

nlp = spacy.load('en_core_web_lg')

TOTAL_THREADS = 15


typos = {}
for line in open("typos.csv"):
    typoLine = line.strip().lower()
    typoLine = typoLine.split(",")
    typos[typoLine[0]] = typoLine[1]

def cleanupAndParseSourceDefinition(input):
    input = input.lower()
    parsed = nlp(input)

    tokensCorrected = []
    hasChanged = False
    for token in parsed:
        newToken = token.text
        if token.text in typos:
            newToken = typos[token.text]
            hasChanged = True

        tokensCorrected += [newToken + token.whitespace_]

    if hasChanged:
        # print("TYPO CHANGE! FROM: '" + input + "' TO '" + "".join(tokensCorrected) + "'")
        parsed = nlp("".join(tokensCorrected))

    return parsed


def getTokensDict(input):
    currentTokens = []

    parsed = cleanupAndParseSourceDefinition(input)

    for token in parsed:
        currentTokens += [{
            "txt": token.text,
            "lem": token.lemma_,
            "ws": token.whitespace_,
            "dep": token.dep_,
            "shape": token.shape_,
            "alpha": token.is_alpha,
            "stop": token.is_stop,
            "pos": token.pos_,
            "tag": token.tag_
        }]

    return currentTokens





def parse(line):
    definitionMeta = json.loads(line)

    definitionMeta["tokens"] = getTokensDict(definitionMeta["word"])

    for definition in definitionMeta["definitions"]:
        definition["tokens"] = getTokensDict(definition["d"])

    return json.dumps(definitionMeta)


lines = []
for line in sys.stdin:
    lines += [line]

with Pool(TOTAL_THREADS) as p:
    lines = p.map(parse, lines)

for line in lines:
    print(line)
