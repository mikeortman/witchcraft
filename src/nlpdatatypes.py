class PartOfSpeech:
    def __init__(self, from_protobuf):
        pass

    def toDict(self):
        pass

class Word:
    def __init__(self, from_protobuf):
        pass

    def getWordString(self):
        pass

    def getPartOfSpeech(self):
        pass

    def getLemmaString(self):
        pass

    def isStopWord(self):
        pass

    def getPostWordWhitespace(self):
        pass

    def getShape(self):
        pass

    def isAlphaWord(self):
        pass

    def toDict(self):
        pass

class Phrase:
    def __init(self):
        pass

    def getWordGenerator(self):
        pass

class Sentence:
    def __init__(self, from_protobuf):
        pass

    def getWordGenerator(self):
        pass

    def toString(self):
        pass

    def toDict(self):
        pass

class SentenceSequence:
    def __init__(self, from_protobuf):
        pass

    def getSentencesGenerator(self):
        pass

    def toDict(self):
        pass

    @staticmethod
    def parseString(input_string : str) -> Sentence:
        return SentenceSequence