# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: projects/naughty/protos/naughty.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from witchcraft.nlp.protos import nlpdatatypes_pb2 as witchcraft_dot_nlp_dot_protos_dot_nlpdatatypes__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='projects/naughty/protos/naughty.proto',
  package='',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n%projects/naughty/protos/naughty.proto\x1a(witchcraft/nlp/protos/nlpdatatypes.proto\"\x9d\x01\n\x19UrbanDictionaryDefinition\x12*\n\x04word\x18\x01 \x02(\x0b\x32\x1c.witchcraft.SentenceSequence\x12\x30\n\ndefinition\x18\x02 \x02(\x0b\x32\x1c.witchcraft.SentenceSequence\x12\x0f\n\x07upvotes\x18\x03 \x02(\x05\x12\x11\n\tdownvotes\x18\x04 \x02(\x05')
  ,
  dependencies=[witchcraft_dot_nlp_dot_protos_dot_nlpdatatypes__pb2.DESCRIPTOR,])




_URBANDICTIONARYDEFINITION = _descriptor.Descriptor(
  name='UrbanDictionaryDefinition',
  full_name='UrbanDictionaryDefinition',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='word', full_name='UrbanDictionaryDefinition.word', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='definition', full_name='UrbanDictionaryDefinition.definition', index=1,
      number=2, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='upvotes', full_name='UrbanDictionaryDefinition.upvotes', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='downvotes', full_name='UrbanDictionaryDefinition.downvotes', index=3,
      number=4, type=5, cpp_type=1, label=2,
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
  serialized_start=84,
  serialized_end=241,
)

_URBANDICTIONARYDEFINITION.fields_by_name['word'].message_type = witchcraft_dot_nlp_dot_protos_dot_nlpdatatypes__pb2._SENTENCESEQUENCE
_URBANDICTIONARYDEFINITION.fields_by_name['definition'].message_type = witchcraft_dot_nlp_dot_protos_dot_nlpdatatypes__pb2._SENTENCESEQUENCE
DESCRIPTOR.message_types_by_name['UrbanDictionaryDefinition'] = _URBANDICTIONARYDEFINITION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

UrbanDictionaryDefinition = _reflection.GeneratedProtocolMessageType('UrbanDictionaryDefinition', (_message.Message,), dict(
  DESCRIPTOR = _URBANDICTIONARYDEFINITION,
  __module__ = 'projects.naughty.protos.naughty_pb2'
  # @@protoc_insertion_point(class_scope:UrbanDictionaryDefinition)
  ))
_sym_db.RegisterMessage(UrbanDictionaryDefinition)


# @@protoc_insertion_point(module_scope)