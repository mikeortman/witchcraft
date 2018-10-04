from typing import Generator, BinaryIO


def protobufs_from_filestream(file: BinaryIO) -> Generator[bytes, None, None]:
    if file.closed or not file.readable():
        return

    while True:
        int_bytes: bytes = file.read(4)
        if len(int_bytes) != 4:
            break

        str_len: int = int.from_bytes(bytes=int_bytes, byteorder="big")
        if str_len < 0:
            break

        raw_message_bytes = file.read(str_len)
        if len(raw_message_bytes) != str_len:
            break

        yield raw_message_bytes


def protobuf_to_filestream(file: BinaryIO, protostr: bytes) -> None:
    if file.closed or not file.writable():
        return

    str_len: int = len(protostr)
    int_bytes: bytes = str_len.to_bytes(4, byteorder="big")
    file.write(int_bytes)
    file.write(protostr)
    file.flush()
