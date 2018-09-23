# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nlpdatatypes.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nlpdatatypes.proto',
  package='witchcraft',
  syntax='proto2',
  serialized_pb=_b('\n\x12nlpdatatypes.proto\x12\nwitchcraft\"\x1b\n\x0cPartOfSpeech\x12\x0b\n\x03pos\x18\x01 \x02(\t\"\xa3\x01\n\x04Word\x12.\n\x0cpartOfSpeech\x18\x01 \x02(\x0b\x32\x18.witchcraft.PartOfSpeech\x12\x0c\n\x04word\x18\x02 \x02(\t\x12\r\n\x05lemma\x18\x03 \x02(\t\x12\x12\n\nisStopWord\x18\x04 \x02(\x08\x12\r\n\x05shape\x18\x05 \x02(\t\x12\x16\n\x0epostWhitespace\x18\x06 \x02(\t\x12\x13\n\x0bisAlphaWord\x18\x07 \x02(\x08\")\n\x06Phrase\x12\x1f\n\x05words\x18\x01 \x03(\x0b\x32\x10.witchcraft.Word\"/\n\x08Sentence\x12#\n\x07phrases\x18\x01 \x03(\x0b\x32\x12.witchcraft.Phrase\";\n\x10SentenceSequence\x12\'\n\tsentences\x18\x01 \x03(\x0b\x32\x14.witchcraft.Sentence')
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
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=34,
  serialized_end=61,
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
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='word', full_name='witchcraft.Word.word', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lemma', full_name='witchcraft.Word.lemma', index=2,
      number=3, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='isStopWord', full_name='witchcraft.Word.isStopWord', index=3,
      number=4, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='witchcraft.Word.shape', index=4,
      number=5, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='postWhitespace', full_name='witchcraft.Word.postWhitespace', index=5,
      number=6, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='isAlphaWord', full_name='witchcraft.Word.isAlphaWord', index=6,
      number=7, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=64,
  serialized_end=227,
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
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=229,
  serialized_end=270,
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
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=272,
  serialized_end=319,
)


_SENTENCESEQUENCE = _descriptor.Descriptor(
  name='SentenceSequence',
  full_name='witchcraft.SentenceSequence',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sentences', full_name='witchcraft.SentenceSequence.sentences', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=321,
  serialized_end=380,
)

_WORD.fields_by_name['partOfSpeech'].message_type = _PARTOFSPEECH
_PHRASE.fields_by_name['words'].message_type = _WORD
_SENTENCE.fields_by_name['phrases'].message_type = _PHRASE
_SENTENCESEQUENCE.fields_by_name['sentences'].message_type = _SENTENCE
DESCRIPTOR.message_types_by_name['PartOfSpeech'] = _PARTOFSPEECH
DESCRIPTOR.message_types_by_name['Word'] = _WORD
DESCRIPTOR.message_types_by_name['Phrase'] = _PHRASE
DESCRIPTOR.message_types_by_name['Sentence'] = _SENTENCE
DESCRIPTOR.message_types_by_name['SentenceSequence'] = _SENTENCESEQUENCE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PartOfSpeech = _reflection.GeneratedProtocolMessageType('PartOfSpeech', (_message.Message,), dict(
  DESCRIPTOR = _PARTOFSPEECH,
  __module__ = 'nlpdatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.PartOfSpeech)
  ))
_sym_db.RegisterMessage(PartOfSpeech)

Word = _reflection.GeneratedProtocolMessageType('Word', (_message.Message,), dict(
  DESCRIPTOR = _WORD,
  __module__ = 'nlpdatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.Word)
  ))
_sym_db.RegisterMessage(Word)

Phrase = _reflection.GeneratedProtocolMessageType('Phrase', (_message.Message,), dict(
  DESCRIPTOR = _PHRASE,
  __module__ = 'nlpdatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.Phrase)
  ))
_sym_db.RegisterMessage(Phrase)

Sentence = _reflection.GeneratedProtocolMessageType('Sentence', (_message.Message,), dict(
  DESCRIPTOR = _SENTENCE,
  __module__ = 'nlpdatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.Sentence)
  ))
_sym_db.RegisterMessage(Sentence)

SentenceSequence = _reflection.GeneratedProtocolMessageType('SentenceSequence', (_message.Message,), dict(
  DESCRIPTOR = _SENTENCESEQUENCE,
  __module__ = 'nlpdatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.SentenceSequence)
  ))
_sym_db.RegisterMessage(SentenceSequence)


# @@protoc_insertion_point(module_scope)
