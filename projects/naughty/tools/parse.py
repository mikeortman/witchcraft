import sys
from projects.naughty.definition import generate_definitions_from_csv
from witchcraft.util.protobuf import protobuf_to_filestream

MAX_PROTOBUF_PER_FILE = 500

i = 0
filenum = 0
fout = open('urban_' + str(filenum) + '.protobuf', 'wb')

for definition in generate_definitions_from_csv(iter(sys.stdin.readline, '')):
    pb = definition.to_protobuf()
    protobuf_to_filestream(fout, pb.SerializeToString())

    i += 1
    if i % MAX_PROTOBUF_PER_FILE == 0:
        print("Done: " + str(i))
        filenum += 1
        fout.close()
        fout = open('urban_' + str(filenum) + '.protobuf', 'wb')
