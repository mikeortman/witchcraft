from witchcraft.nlp.protos.nlpdatatypes_pb2 import Word as WordProto
from witchcraft.nlp.partofspeech import PartOfSpeech

class Word:
    def __init__(self):
        self._word = ''
        self._pos = PartOfSpeech()
        self._lemma = ''
        self._is_stop_word = False
        self._whitespace_postfix = ''
        self._shape = ''
        self._is_alpha_word = False

    def get_word_string(self) -> str:
        return self._word

    def get_part_of_speech(self) -> PartOfSpeech:
        return self._pos

    def get_lemma_string(self) -> str:
        return self._lemma

    def is_stop_word(self) -> bool:
        return self._is_stop_word

    def get_whitespace_postfix(self) -> str:
        return self._whitespace_postfix

    def get_shape(self) -> str:
        return self._shape

    def is_alpha_word(self) -> bool:
        return self._is_alpha_word

    def to_protobuf(self) -> WordProto:
        return WordProto(
            word=self.get_word_string(),
            partOfSpeech=self.get_part_of_speech().to_protobuf(),
            lemma=self.get_lemma_string(),
            isStopWord=self.is_stop_word(),
            postWhitespace=self.get_whitespace_postfix(),
            shape=self.get_shape(),
            isAlphaWord=self.is_alpha_word()
        )
