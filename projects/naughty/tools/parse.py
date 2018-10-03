import sys
from google.protobuf.json_format import MessageToJson
from projects.naughty.definition import UrbanDictionaryDefinition

i = 0
for definition in UrbanDictionaryDefinition.generate_from_csv(iter(sys.stdin.readline, '')):
    pb = definition.to_protobuf()
    i += 1
    if i % 100 == 0:
        print ("Done: " + str(i))
