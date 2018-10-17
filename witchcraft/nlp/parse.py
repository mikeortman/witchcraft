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
    def cached_sentence_sequence_generator(src_generator):
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
    def cluster_phrases(sentence_sequences: Callable[[], Generator[SentenceSequence, None, None]], max_ngram: int, min_appear: int) -> Generator[SentenceSequence, None, None]:
        # This method does phrase clustering up to max_ngram sizes, starting at largest ngram.
        # Requires an ngram to appear min_appear times before it would cluster on that.
        # First pass will count frequency of adjacent phrases
        # Second pass will do the grouping on a first-come (LTR) basis on phrases meeting requirements
        # Not recommended for very large corpuses right now, but good for most.
        # Not recommended for multiple times unnecessarily
        def ngramed_sequence_gen(sequence_gen, ngram_size):
            def gen():
                phrase_combo_frequency = {}
                for sentence_sequence in sequence_gen():
                    for sentence in sentence_sequence.get_sentence_generator():
                        phrases = list(sentence.get_phrase_generator())
                        for x in range(len(phrases) - ngram_size + 1):
                            phrases_current_ngram = phrases[x:x+ngram_size]
                            passing_phrases = [p for p in phrases_current_ngram if p.provides_contextual_value() and p.is_word()]
                            if len(phrases_current_ngram) != len(passing_phrases):
                                continue  # Not all items provide contextual value

                            combo = Phrase.merge_phrases(phrases_current_ngram).to_phrase_normalized()
                            if combo not in phrase_combo_frequency:
                                phrase_combo_frequency[combo] = 0

                            phrase_combo_frequency[combo] += 1

                phrases_meeting_requirements = {k: v for k, v in phrase_combo_frequency.items() if v >= min_appear}
                print(phrases_meeting_requirements)

                # Second pass, do the grouping.
                for sentence_sequence in sequence_gen():
                    new_sequence_sentences = []
                    for sentence in sentence_sequence.get_sentence_generator():
                        # print("Running phrase merge on " + str(sentence))
                        phrases = list(sentence.get_phrase_generator())
                        # print ("Got phrases: " + str(phrases))

                        merged_phrases = []
                        x = 0
                        while x < len(phrases) - ngram_size + 1:
                            phrases_current_ngram = phrases[x:x+ngram_size]
                            # print("Testing on " + str([p.to_phrase_normalized() for p in phrases_current_ngram]))

                            passing_phrases = [p for p in phrases_current_ngram if p.provides_contextual_value() and p.is_word()]
                            if len(phrases_current_ngram) != len(passing_phrases):
                                merged_phrases += [phrases_current_ngram[0]]
                                x += 1
                                continue

                            combo = Phrase.merge_phrases(phrases_current_ngram)
                            combo_str = combo.to_phrase_normalized()
                            if combo_str in phrases_meeting_requirements:
                                merged_phrases += [combo]
                                x += ngram_size
                            else:
                                merged_phrases += [phrases_current_ngram[0]]
                                x += 1

                        # Edge case where the last ngram will not become a phrase
                        if x == len(phrases) - ngram_size + 1:
                            # print("TEST EDGE CASE: " + str(x) + "; LEN: " + str(len(phrases)) + "; NGRAM: " + str(ngram_size))
                            merged_phrases += phrases[x:]

                        new_sentence = Sentence(merged_phrases)
                        new_sequence_sentences += [new_sentence]

                    # print("Yielding: " + str(new_sequence_sentences))
                    yield SentenceSequence(new_sequence_sentences)
            return gen

        current_sequence_gen = sentence_sequences
        for i in range(max_ngram-1): #3 -> 0,1. Runs gen on 3, then 2
            current_sequence_gen = ngramed_sequence_gen(current_sequence_gen, max_ngram - i)
            current_sequence_gen = Parser.cached_sentence_sequence_generator(current_sequence_gen)

        return current_sequence_gen