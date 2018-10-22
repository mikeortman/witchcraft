# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: witchcraft/nlp/protos/nlpdatatypes.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='witchcraft/nlp/protos/nlpdatatypes.proto',
  package='witchcraft',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n(witchcraft/nlp/protos/nlpdatatypes.proto\x12\nwitchcraft\"\x1b\n\x0cPartOfSpeech\x12\x0b\n\x03pos\x18\x01 \x02(\t\"A\n\x0eWordDependency\x12\x0b\n\x03\x64\x65p\x18\x01 \x02(\t\x12\x11\n\theadIndex\x18\x02 \x02(\r\x12\x0f\n\x07myIndex\x18\x03 \x02(\r\"\xd3\x01\n\x04Word\x12.\n\x0cpartOfSpeech\x18\x01 \x02(\x0b\x32\x18.witchcraft.PartOfSpeech\x12\x0c\n\x04word\x18\x02 \x02(\t\x12\r\n\x05lemma\x18\x03 \x02(\t\x12\x12\n\nisStopWord\x18\x04 \x02(\x08\x12\r\n\x05shape\x18\x05 \x02(\t\x12\x16\n\x0epostWhitespace\x18\x06 \x02(\t\x12\x13\n\x0bisAlphaWord\x18\x07 \x02(\x08\x12.\n\ndependency\x18\x08 \x02(\x0b\x32\x1a.witchcraft.WordDependency\")\n\x06Phrase\x12\x1f\n\x05words\x18\x01 \x03(\x0b\x32\x10.witchcraft.Word\"/\n\x08Sentence\x12#\n\x07phrases\x18\x01 \x03(\x0b\x32\x12.witchcraft.Phrase\"3\n\x08\x44ocument\x12\'\n\tsentences\x18\x01 \x03(\x0b\x32\x14.witchcraft.Sentence\"6\n\rWordEmbedding\x12\x0c\n\x04word\x18\x01 \x02(\t\x12\x17\n\x0f\x65mbeddingVector\x18\x02 \x03(\x01')
)




_PARTOFSPEECH = _descriptor.Descriptor(
  name='PartOfSpeech',
  full_name='witchcraft.PartOfSpeech',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pos', full_name='witchcraft.PartOfSpeech.pos', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=56,
  serialized_end=83,
)


_WORDDEPENDENCY = _descriptor.Descriptor(
  name='WordDependency',
  full_name='witchcraft.WordDependency',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dep', full_name='witchcraft.WordDependency.dep', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='headIndex', full_name='witchcraft.WordDependency.headIndex', index=1,
      number=2, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='myIndex', full_name='witchcraft.WordDependency.myIndex', index=2,
      number=3, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=85,
  serialized_end=150,
)


_WORD = _descriptor.Descriptor(
  name='Word',
  full_name='witchcraft.Word',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='partOfSpeech', full_name='witchcraft.Word.partOfSpeech', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='word', full_name='witchcraft.Word.word', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lemma', full_name='witchcraft.Word.lemma', index=2,
      number=3, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='isStopWord', full_name='witchcraft.Word.isStopWord', index=3,
      number=4, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='witchcraft.Word.shape', index=4,
      number=5, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='postWhitespace', full_name='witchcraft.Word.postWhitespace', index=5,
      number=6, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='isAlphaWord', full_name='witchcraft.Word.isAlphaWord', index=6,
      number=7, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dependency', full_name='witchcraft.Word.dependency', index=7,
      number=8, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=153,
  serialized_end=364,
)


_PHRASE = _descriptor.Descriptor(
  name='Phrase',
  full_name='witchcraft.Phrase',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='words', full_name='witchcraft.Phrase.words', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=366,
  serialized_end=407,
)


_SENTENCE = _descriptor.Descriptor(
  name='Sentence',
  full_name='witchcraft.Sentence',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='phrases', full_name='witchcraft.Sentence.phrases', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=409,
  serialized_end=456,
)


_DOCUMENT = _descriptor.Descriptor(
  name='Document',
  full_name='witchcraft.Document',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sentences', full_name='witchcraft.Document.sentences', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=458,
  serialized_end=509,
)


_WORDEMBEDDING = _descriptor.Descriptor(
  name='WordEmbedding',
  full_name='witchcraft.WordEmbedding',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='word', full_name='witchcraft.WordEmbedding.word', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='embeddingVector', full_name='witchcraft.WordEmbedding.embeddingVector', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=511,
  serialized_end=565,
)

_WORD.fields_by_name['partOfSpeech'].message_type = _PARTOFSPEECH
_WORD.fields_by_name['dependency'].message_type = _WORDDEPENDENCY
_PHRASE.fields_by_name['words'].message_type = _WORD
_SENTENCE.fields_by_name['phrases'].message_type = _PHRASE
_DOCUMENT.fields_by_name['sentences'].message_type = _SENTENCE
DESCRIPTOR.message_types_by_name['PartOfSpeech'] = _PARTOFSPEECH
DESCRIPTOR.message_types_by_name['WordDependency'] = _WORDDEPENDENCY
DESCRIPTOR.message_types_by_name['Word'] = _WORD
DESCRIPTOR.message_types_by_name['Phrase'] = _PHRASE
DESCRIPTOR.message_types_by_name['Sentence'] = _SENTENCE
DESCRIPTOR.message_types_by_name['Document'] = _DOCUMENT
DESCRIPTOR.message_types_by_name['WordEmbedding'] = _WORDEMBEDDING
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PartOfSpeech = _reflection.GeneratedProtocolMessageType('PartOfSpeech', (_message.Message,), dict(
  DESCRIPTOR = _PARTOFSPEECH,
  __module__ = 'witchcraft.nlp.protos.nlpdatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.PartOfSpeech)
  ))
_sym_db.RegisterMessage(PartOfSpeech)

WordDependency = _reflection.GeneratedProtocolMessageType('WordDependency', (_message.Message,), dict(
  DESCRIPTOR = _WORDDEPENDENCY,
  __module__ = 'witchcraft.nlp.protos.nlpdatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.WordDependency)
  ))
_sym_db.RegisterMessage(WordDependency)

Word = _reflection.GeneratedProtocolMessageType('Word', (_message.Message,), dict(
  DESCRIPTOR = _WORD,
  __module__ = 'witchcraft.nlp.protos.nlpdatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.Word)
  ))
_sym_db.RegisterMessage(Word)

Phrase = _reflection.GeneratedProtocolMessageType('Phrase', (_message.Message,), dict(
  DESCRIPTOR = _PHRASE,
  __module__ = 'witchcraft.nlp.protos.nlpdatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.Phrase)
  ))
_sym_db.RegisterMessage(Phrase)

Sentence = _reflection.GeneratedProtocolMessageType('Sentence', (_message.Message,), dict(
  DESCRIPTOR = _SENTENCE,
  __module__ = 'witchcraft.nlp.protos.nlpdatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.Sentence)
  ))
_sym_db.RegisterMessage(Sentence)

Document = _reflection.GeneratedProtocolMessageType('Document', (_message.Message,), dict(
  DESCRIPTOR = _DOCUMENT,
  __module__ = 'witchcraft.nlp.protos.nlpdatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.Document)
  ))
_sym_db.RegisterMessage(Document)

WordEmbedding = _reflection.GeneratedProtocolMessageType('WordEmbedding', (_message.Message,), dict(
  DESCRIPTOR = _WORDEMBEDDING,
  __module__ = 'witchcraft.nlp.protos.nlpdatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.WordEmbedding)
  ))
_sym_db.RegisterMessage(WordEmbedding)


# @@protoc_insertion_point(module_scope)
