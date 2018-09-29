from typing import Optional, List
import spacy

from witchcraft.nlp.datatypes import SentenceSequence, Sentence, Phrase, Word, PartOfSpeech, WordDependency

class SpacyInstance:
    __instance: Optional['SpacyInstance'] = None

    def __init__(self) -> None:
        self._nlp = spacy.load('en_core_web_lg')

    def parse_internal(self, input_str: str):
        return self._nlp(input_str)


    @classmethod
    def parse(cls, input_str: str):
        if not cls.__instance:
            cls.__instance = SpacyInstance()

        return cls.__instance.parse_internal(input_str)


class Parser:
    @staticmethod
    def parse_string_to_sentence_sequence(input_str: str) -> SentenceSequence:
        sentences: List[Sentence] = []
        for src_sentence in SpacyInstance.parse(input_str).sents:
            phrases: List[Phrase] = []
            for src_token in src_sentence:
                word_dependency: WordDependency = WordDependency(
                    dep=src_token.dep_,
                    my_index=src_token.i - src_sentence.start,
                    head_index=src_token.head.i - src_sentence.start
                )

                word: Word = Word(
                    word=src_token.text,
                    lemma=src_token.lemma_,
                    pos=PartOfSpeech(src_token.pos_),
                    is_stop_word=src_token.is_stop,
                    whitespace_postfix=src_token.whitespace_,
                    shape=src_token.shape_,
                    is_alpha_word=src_token.is_alpha,
                    word_dependency=word_dependency
                )

                phrase: Phrase = Phrase([word])
                phrases += [phrase]

            sentence: Sentence = Sentence(phrases=phrases)
            sentences += [sentence]

        return SentenceSequence(sentences=sentences)
