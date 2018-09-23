from nlp.protos.nlpdatatypes_pb2 import Word as WordProto
from nlp.partofspeech import PartOfSpeech

class Word:
    def __init__(self, from_protobuf: WordProto):
        from_protobuf.g
        pass

    def getWordString(self) -> str:
        pass

    def getPartOfSpeech(self) -> PartOfSpeech:
        pass

    def getLemmaString(self) -> str:
        pass

    def isStopWord(self) -> bool:
        pass

    def getPostWordWhitespace(self) -> str:
        pass

    def getShape(self) -> str:
        pass

    def isAlphaWord(self) -> bool:
        pass
