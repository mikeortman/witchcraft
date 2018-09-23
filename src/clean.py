import sys
import csv
import re
import json
from string import punctuation


def cleanOutHumanShit(strin):
	#Remove letters that appear more than twice
	if len(strin) > 3:
		newStr = [strin[0], strin[1], strin[2]]

		for i in range(3, len(strin)):
			greatGrandParent = strin[i - 3]
			grandParent = strin[i - 2]
			parent = strin[i - 1]
			child = strin[i]

			if greatGrandParent is child and grandParent is child and parent is  child:
				continue

			newStr += [child]


		strin = ''.join(newStr)

	strin = re.sub(r"^(?:#?[0-9a-z]{1,2} *(?:->|=>|-|\.|\)|,)+|\-+) *", "", strin) # Remove list ordinals.
	strin = re.sub(r"[\\[\]]", "", strin) # Remove brackets.
	strin = re.sub(r"\\*", "", strin) # Remove asteriks
	strin = re.sub(r"\\~", "", strin) # Remove tildes
	strin = re.sub(r"[-,.!? ]+$", "", strin) # Remove end of line punctuation.
	strin = re.sub(r"^(?:[(\[]?(?:noun|n|verb|v|adverb|adv|av|adjective|adj|aj)[\.,;]*[)\].]*[^a-z\n])+", "", strin) # Remove pos markers
	strin = strin.strip()

	return strin



allDefinitions = {}

for row in csv.reader(iter(sys.stdin.readline, '')):
	if len(row) < 6:
		continue

	upvotes = 0
	downvotes = 0

	try:
		upvotes = int(row[2])
		downvotes = int(row[3])
	except ValueError:
		pass  # it was a string, not an int.

	totalVotes = upvotes + downvotes
	effectiveVote = upvotes - downvotes

	if totalVotes < 50 or effectiveVote < 40:
		continue


	word = row[1].strip().lower().strip(punctuation)
	definition = row[5].strip().lower()

	if(len(word) < 3 or len(word) > 32):
		continue

	if not all(ord(c) < 128 for c in word):
		continue


	if not all(ord(c) < 128 for c in definition):
		continue

	if ('<i>' in definition or '<b>' in definition or '<strong>' in definition):
		continue

	if word in definition:
		continue

	word = cleanOutHumanShit(word)
	definitionLines = definition.split(';;')
	definitionLines = [cleanOutHumanShit(d) for d in definitionLines]
	definition = '\n'.join([d for d in definitionLines if len(d) > 0])


	if word not in allDefinitions:
		allDefinitions[word] = []

	allDefinitions[word] += [[word, effectiveVote, totalVotes, definition]]

allWordTitles = allDefinitions.keys()
sortedWordTitles = sorted(allDefinitions)


for word in sortedWordTitles:
	row = allDefinitions[word]
	orderedRow = sorted(row, key = lambda r: r[1])

	curJson = {
		"word": word,
		"definitions": [{
			"ev": d[1],
			"tv": d[2],
            "d": d[3]
		} for d in orderedRow]
	}

	print (json.dumps(curJson))

print ("Total Words: " + str(len(sortedWordTitles)))
