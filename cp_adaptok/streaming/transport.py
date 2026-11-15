import struct

from .bitstream import MAX_RECORD_BYTES, StreamError


def send_record(connection, data):
    if not 0 < len(data) <= MAX_RECORD_BYTES:
        raise StreamError("invalid transport record size")
    connection.sendall(struct.pack(">I", len(data)) + data)
    return 4 + len(data)


def receive_exact(connection, count):
    buffer = bytearray()
    while len(buffer) < count:
        part = connection.recv(count - len(buffer))
        if not part:
            raise EOFError("socket closed before complete record")
        buffer.extend(part)
    return bytes(buffer)


def receive_record(connection):
    size = struct.unpack(">I", receive_exact(connection, 4))[0]
    if not 0 < size <= MAX_RECORD_BYTES:
        raise StreamError("transport record exceeds allocation limit")
    return receive_exact(connection, size)
