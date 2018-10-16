from typing import Optional, List, Generator, Callable
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
    def create_cached_sentence_sequence_generator(src_generator):
        cached = list(src_generator())

        def cached_gen():
            i = 0
            for sequence in cached:
                yield sequence

                i += 1
                if i % 5000 == 0:
                    print ("Retreived " + str(i) + " from cache.")

        return cached_gen

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

    @staticmethod
    def cluster_phrases_with_minimum_appearance(sentence_sequences: Callable[[], Generator[SentenceSequence, None, None]], min_appearances:int) -> Generator[SentenceSequence, None, None]:
        # This method does phrase cluster.
        # First pass will count frequency of adjacent phrases
        # Second pass will do the grouping on a first-come (LTR) basis on phrases meeting requirements
        # Not recommended for very large corpuses right now, but good for most.
        # Not recommended for multiple times unnecessarily
        def gen():
            phrase_combo_frequency = {}

            for sentence_sequence in sentence_sequences():
                for sentence in sentence_sequence.get_sentence_generator():
                    phrases = list(sentence.get_phrase_generator())
                    for i in range(len(phrases) - 1):
                        first = phrases[i]
                        second = phrases[i+1]

                        if not first.provides_contextual_value() or not second.provides_contextual_value():
                            continue

                        combo = Phrase.merge_two_phrases(first, second).to_phrase_normalized()
                        if combo not in phrase_combo_frequency:
                            phrase_combo_frequency[combo] = 0

                        phrase_combo_frequency[combo] += 1

            phrases_meeting_requirements = {k: v for k, v in phrase_combo_frequency.items() if v >= min_appearances}

            print(phrases_meeting_requirements)

            # Second pass, do the grouping.
            for sentence_sequence in sentence_sequences():
                new_sequence_sentences = []
                for sentence in sentence_sequence.get_sentence_generator():
                    # print("Running phrase merge on " + str(sentence))
                    phrases = list(sentence.get_phrase_generator())
                    # print ("Got phrases: " + str(phrases))

                    merged_phrases = []
                    i = 0
                    while i < len(phrases) - 1:
                        first = phrases[i]
                        second = phrases[i+1]
                        # print("Testing on " + str(first) + " and " + str(second))

                        if not first.provides_contextual_value() or not second.provides_contextual_value():
                            merged_phrases += [first]
                            i += 1
                            continue

                        merged_phrase = Phrase.merge_two_phrases(first, second)
                        combo_str = merged_phrase.to_phrase_normalized()
                        if combo_str in phrases_meeting_requirements:
                            merged_phrases += [merged_phrase]
                            i += 2
                        else:
                            merged_phrases += [first]
                            i += 1

                    # Edge case where the last two words will not become a phrase
                    if i == len(phrases) - 1:
                        merged_phrases += [phrases[i]]

                    new_sentence = Sentence(merged_phrases)
                    # print("Merged: " + str(merged_phrases))
                    new_sequence_sentences += [new_sentence]

                # print("Yielding: " + str(new_sequence_sentences))
                yield SentenceSequence(new_sequence_sentences)
        return gen