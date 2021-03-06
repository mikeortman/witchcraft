# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: witchcraft/ml/protos/mldatatypes.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='witchcraft/ml/protos/mldatatypes.proto',
  package='witchcraft',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n&witchcraft/ml/protos/mldatatypes.proto\x12\nwitchcraft\"%\n\tEmbedding\x12\x18\n\x10\x65mbedding_vector\x18\x02 \x03(\x01\"8\n\x14PhraseEmbeddingNgram\x12\r\n\x05ngram\x18\x01 \x02(\t\x12\x11\n\tattention\x18\x02 \x01(\x02\"\x8c\x01\n\x0fPhraseEmbedding\x12(\n\tembedding\x18\x01 \x02(\x0b\x32\x15.witchcraft.Embedding\x12\x0e\n\x06phrase\x18\x02 \x02(\t\x12\r\n\x05\x63ount\x18\x03 \x02(\x05\x12\x30\n\x06ngrams\x18\x04 \x03(\x0b\x32 .witchcraft.PhraseEmbeddingNgram')
)




_EMBEDDING = _descriptor.Descriptor(
  name='Embedding',
  full_name='witchcraft.Embedding',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='embedding_vector', full_name='witchcraft.Embedding.embedding_vector', index=0,
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
  serialized_start=54,
  serialized_end=91,
)


_PHRASEEMBEDDINGNGRAM = _descriptor.Descriptor(
  name='PhraseEmbeddingNgram',
  full_name='witchcraft.PhraseEmbeddingNgram',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ngram', full_name='witchcraft.PhraseEmbeddingNgram.ngram', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attention', full_name='witchcraft.PhraseEmbeddingNgram.attention', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=93,
  serialized_end=149,
)


_PHRASEEMBEDDING = _descriptor.Descriptor(
  name='PhraseEmbedding',
  full_name='witchcraft.PhraseEmbedding',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='embedding', full_name='witchcraft.PhraseEmbedding.embedding', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='phrase', full_name='witchcraft.PhraseEmbedding.phrase', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='count', full_name='witchcraft.PhraseEmbedding.count', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ngrams', full_name='witchcraft.PhraseEmbedding.ngrams', index=3,
      number=4, type=11, cpp_type=10, label=3,
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
  serialized_start=152,
  serialized_end=292,
)

_PHRASEEMBEDDING.fields_by_name['embedding'].message_type = _EMBEDDING
_PHRASEEMBEDDING.fields_by_name['ngrams'].message_type = _PHRASEEMBEDDINGNGRAM
DESCRIPTOR.message_types_by_name['Embedding'] = _EMBEDDING
DESCRIPTOR.message_types_by_name['PhraseEmbeddingNgram'] = _PHRASEEMBEDDINGNGRAM
DESCRIPTOR.message_types_by_name['PhraseEmbedding'] = _PHRASEEMBEDDING
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Embedding = _reflection.GeneratedProtocolMessageType('Embedding', (_message.Message,), dict(
  DESCRIPTOR = _EMBEDDING,
  __module__ = 'witchcraft.ml.protos.mldatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.Embedding)
  ))
_sym_db.RegisterMessage(Embedding)

PhraseEmbeddingNgram = _reflection.GeneratedProtocolMessageType('PhraseEmbeddingNgram', (_message.Message,), dict(
  DESCRIPTOR = _PHRASEEMBEDDINGNGRAM,
  __module__ = 'witchcraft.ml.protos.mldatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.PhraseEmbeddingNgram)
  ))
_sym_db.RegisterMessage(PhraseEmbeddingNgram)

PhraseEmbedding = _reflection.GeneratedProtocolMessageType('PhraseEmbedding', (_message.Message,), dict(
  DESCRIPTOR = _PHRASEEMBEDDING,
  __module__ = 'witchcraft.ml.protos.mldatatypes_pb2'
  # @@protoc_insertion_point(class_scope:witchcraft.PhraseEmbedding)
  ))
_sym_db.RegisterMessage(PhraseEmbedding)


# @@protoc_insertion_point(module_scope)
