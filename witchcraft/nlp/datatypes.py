from typing import Generator, List, Optional, Tuple

from witchcraft.nlp.protos.nlpdatatypes_pb2 import Word as WordProto
from witchcraft.nlp.protos.nlpdatatypes_pb2 import PartOfSpeech as PartOfSpeechProto
from witchcraft.nlp.protos.nlpdatatypes_pb2 import Phrase as PhraseProto
from witchcraft.nlp.protos.nlpdatatypes_pb2 import Sentence as SentenceProto
from witchcraft.nlp.protos.nlpdatatypes_pb2 import Document as DocumentProto
from witchcraft.nlp.protos.nlpdatatypes_pb2 import WordDependency as WordDependencyProto
from witchcraft.util.hash import hash_fnv1a
from witchcraft.util.listutil import skipgramify


class PartOfSpeech:
    def __init__(self, pos: Optional[str] = None) -> None:
        self._pos = pos if pos is not None else 'UNK'

    def __str__(self) -> str:
        return self._pos

    def to_protobuf(self) -> PartOfSpeechProto:
        return PartOfSpeechProto(
            pos=self._pos
        )

    def to_array(self) -> List[any]:
        return [self._pos]

    def provides_contextual_value(self) -> bool:
        return self._pos not in ['SYM', 'PUNCT', 'SPACE', 'NUM']

    @classmethod
    def from_protobuf(cls, pos_proto: PartOfSpeechProto) -> 'PartOfSpeech':
        return PartOfSpeech(pos_proto.pos)

    @classmethod
    def from_array(cls, arr: List[any]) -> 'PartOfSpeech':
        return PartOfSpeech(
            pos=arr[0]
        )


class WordDependency:
    def __init__(self, my_index: int, head_index: int, dep: str = 'UNK') -> None:
        self._dep: str = dep
        self._head_index: int = head_index
        self._my_index: int = my_index

    def get_dep(self) -> str:
        return self._dep

    def get_head_index(self) -> int:
        return self._head_index

    def get_my_index(self) -> int:
        return self._my_index

    def to_protobuf(self) -> WordDependencyProto:
        return WordDependencyProto(
            dep=self.get_dep(),
            headIndex=self.get_head_index(),
            myIndex=self.get_my_index()
        )

    def to_array(self) -> List[any]:
        return [
            self.get_dep(),
            self.get_head_index(),
            self.get_my_index()
        ]

    @classmethod
    def from_protobuf(cls, dep_proto: WordDependencyProto) -> 'WordDependency':
        return WordDependency(dep=dep_proto.dep, my_index=dep_proto.myIndex, head_index=dep_proto.headIndex)

    @classmethod
    def from_array(cls, arr: List[any]) -> 'WordDependency':
        return WordDependency(
            my_index=arr[2],
            head_index=arr[1],
            dep=arr[0]
        )


class Word:
    def __init__(self,
                 word: str,
                 lemma: Optional[str] = None,
                 pos: Optional[PartOfSpeech] = None,
                 is_stop_word: bool = False,
                 whitespace_postfix: str = '',
                 shape: str = '',
                 is_alpha_word: bool = False,
                 word_dependency: Optional[WordDependency] = None) -> None:
        self._word: str = word
        self._word_normalized = self._word.strip().lower()
        self._word_hash = hash_fnv1a(input_str=self._word_normalized, bucket_size=None)

        self._word_len: int = len(word)
        self._pos: PartOfSpeech = pos if pos is not None else PartOfSpeech()
        self._lemma: str = lemma if lemma is not None else word
        self._is_stop_word: bool = is_stop_word
        self._whitespace_postfix: str = whitespace_postfix
        self._shape: str = shape
        self._is_alpha_word: bool = is_alpha_word
        self._word_dependency: WordDependency = word_dependency if word_dependency is not None else WordDependency(0, 0)

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

    def get_word_dependency(self) -> WordDependency:
        return self._word_dependency

    def provides_contextual_value(self) -> bool:
        return self._pos.provides_contextual_value() and self._word_len > 1 and self._word[0] != "'"

    def get_word_string_normalized(self):
        return self._word_normalized

    def to_protobuf(self) -> WordProto:
        return WordProto(
            word=self.get_word_string(),
            partOfSpeech=self.get_part_of_speech().to_protobuf(),
            lemma=self.get_lemma_string(),
            isStopWord=self.is_stop_word(),
            postWhitespace=self.get_whitespace_postfix(),
            shape=self.get_shape(),
            isAlphaWord=self.is_alpha_word(),
            dependency=self.get_word_dependency().to_protobuf()
        )

    def to_array(self) -> List[any]:
        return [
            self.get_word_string(),
            self.get_part_of_speech().to_array(),
            self.get_lemma_string(),
            self.is_stop_word(),
            self.get_whitespace_postfix(),
            self.get_shape(),
            self.is_alpha_word(),
            self.get_word_dependency().to_array()
        ]

    def ngrams(self, ngram_len: int) -> List[str]:
        word = "<" + self.get_word_string_normalized() + ">"
        ngrams = []
        # chunk into ngrams.
        for i in range(len(word) - ngram_len + 1):
            word += [word[i:i + ngram_len]]

        return ngrams

    def ngram_skipgrams(self, window_size: int, ngram_len: int) -> List[Tuple[str, str, int]]:
        return skipgramify(self.ngrams(ngram_len), window_size)

    @classmethod
    def from_protobuf(cls, word_proto: WordProto) -> 'Word':
        return Word(
            word=word_proto.word,
            lemma=word_proto.lemma,
            pos=PartOfSpeech.from_protobuf(word_proto.partOfSpeech),
            is_stop_word=word_proto.isStopWord,
            is_alpha_word=word_proto.isAlphaWord,
            whitespace_postfix=word_proto.postWhitespace,
            shape=word_proto.shape,
            word_dependency=WordDependency.from_protobuf(word_proto.dependency)
        )

    @classmethod
    def from_array(cls, arr: List[any]) -> 'Word':
        return Word(
            word=arr[0],
            pos=PartOfSpeech.from_array(arr[1]),
            lemma=arr[2],
            is_stop_word=arr[3],
            whitespace_postfix=arr[4],
            shape=arr[5],
            is_alpha_word=arr[6],
            word_dependency=WordDependency.from_array(arr[7])
        )


class Phrase:
    def __init__(self, words: Optional[List[Word]] = None) -> None:
        self._words: List[Word] = words if words is not None else []
        self._phrase_str = ''.join(
            [w.get_word_string() + w.get_whitespace_postfix() for w in self]
        )

        self._phrase_str_norm = ' '.join([w.get_word_string_normalized() for w in self])
        self._phrase_hash = hash_fnv1a(input_str=self._phrase_str_norm, bucket_size=None)

        self._provides_contextual_value = True
        for word in self:
            if not word.provides_contextual_value():
                self._provides_contextual_value = False
                break

    def __iter__(self):
        return iter(self._words)

    def __str__(self)-> str:
        # TODO: Make Faster
        return self._phrase_str

    def to_phrase_normalized(self):
        return self._phrase_str_norm

    def to_protobuf(self) -> PhraseProto:
        return PhraseProto(
            words=[w.to_protobuf() for w in self]
        )

    def get_hash(self, bucket_size: Optional[int] = None):
        if bucket_size is not None:
            return self._phrase_hash % bucket_size

        return self._phrase_hash

    def is_word(self):
        return len(self._words) == 1

    def to_array(self) -> List[any]:
        return [w.to_array() for w in self]

    def provides_contextual_value(self) -> bool:
        return self._provides_contextual_value

    def ngrams(self, ngram_len: int) -> List[str]:
        all_ngrams = []
        for word in self._words:
            all_ngrams += word.ngrams(ngram_len)

        return all_ngrams

    @classmethod
    def from_protobuf(cls, phrase_proto: PhraseProto) -> 'Phrase':
        return Phrase([Word.from_protobuf(word_proto=w) for w in phrase_proto.words])

    @classmethod
    def from_array(cls, arr: List[any]) -> 'Phrase':
        return Phrase(
            words=[Word.from_array(w) for w in arr]
        )

    @classmethod
    def merge_phrases(cls, phrases: List['Phrase']) -> 'Phrase':
        words = []
        for phrase in phrases:
            words += list(iter(phrase))

        return Phrase(words)


class Sentence:
    def __init__(self, phrases: Optional[List[Phrase]] = None) -> None:
        self._phrases: List[Phrase] = phrases if phrases is not None else []

    def __iter__(self):
        return iter(self._phrases)

    def __str__(self) -> str:
        return ''.join([str(p) for p in self])

    def to_protobuf(self) -> SentenceProto:
        return SentenceProto(
            phrases=[p.to_protobuf() for p in self]
        )

    def to_array(self) -> List[any]:
        return [p.to_array() for p in self]

    def skipgrams(self, window_size: int) -> Generator[Tuple[Phrase, Phrase, int], None, None]:
        valid_phrases = [p for p in self._phrases if p.provides_contextual_value()]
        sentence_size: int = len(valid_phrases)

        for i in range(sentence_size):
            current_phrase = valid_phrases[i]
            yield (current_phrase, current_phrase, 0)

            if current_phrase is None:
                continue

            for y in range(1, window_size + 1):
                if i - y >= 0 and valid_phrases[i - y] is not None:
                    yield (current_phrase, valid_phrases[i - y], -y)

            for y in range(1, window_size + 1):
                if i + y < sentence_size and valid_phrases[i + y] is not None:
                    yield (current_phrase, valid_phrases[i + y], y)


    @classmethod
    def from_protobuf(cls, sentence_proto: SentenceProto) -> 'Sentence':
        return Sentence([Phrase.from_protobuf(phrase_proto=p) for p in sentence_proto.phrases])

    @classmethod
    def from_array(cls, arr: List[any]) -> 'Sentence':
        return Sentence(
            phrases=[Phrase.from_array(p) for p in arr]
        )


class Document:
    def __init__(self, sentences: Optional[List[Sentence]] = None) -> None:
        self._sentences: List[Sentence] = sentences if sentences is not None else []

    def __iter__(self):
        return iter(self._sentences)

    def __str__(self) -> str:
        return ''.join([str(s) for s in iter(self)])

    def to_protobuf(self) -> DocumentProto:
        return DocumentProto(
            sentences=[s.to_protobuf() for s in iter(self)]
        )

    def to_array(self) -> List[any]:
        return [s.to_array() for s in iter(self)]

    @classmethod
    def from_protobuf(cls, document_proto: DocumentProto) -> 'Document':
        return Document([Sentence.from_protobuf(sentence_proto=s) for s in document_proto.sentences])

    @classmethod
    def from_array(cls, arr: List[any]) -> 'Document':
        return Document(
            sentences=[Sentence.from_array(s) for s in arr]
        )

class Corpus:
    def __init__(self, documents: Optional[List[Document]] = None) -> None:
        self._documents: List[Document] = documents if documents is not None else []

    def __iter__(self):
        return iter(self._documents)