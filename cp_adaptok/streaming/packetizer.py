from .bitstream import Packet, StreamError, PACKET_HEADER_BYTES


def packetize(header, ids, entropy):
    if len(ids) != header.n_max:
        raise ValueError("encode_full must supply exactly N_max IDs")
    packets = []
    for first in range(0, len(ids), header.packet_tokens):
        block = first // header.packet_tokens
        part = ids[first:first + header.packet_tokens]
        packets.append(Packet(header.frame_id, first, len(part), block, entropy.encode(part, block)).to_bytes())
    return tuple(packets)


class ProgressivePacketStream:
    def __init__(self, header, entropy):
        self.header, self.entropy = header, entropy
        self.token_ids = []
        self.packets = []

    def append(self, wire):
        packet = Packet.from_bytes(wire)
        h = self.header
        if packet.frame_id != h.frame_id:
            raise StreamError("packet belongs to a different frame")
        if packet.first_token_index != len(self.token_ids):
            raise StreamError("packet must extend the contiguous prefix; missing, duplicate or out-of-order packet")
        if packet.block_id != len(self.packets) or packet.token_count != h.packet_tokens:
            raise StreamError("invalid packet block/count")
        if len(self.token_ids) + packet.token_count > h.n_max:
            raise StreamError("prefix exceeds N_max")
        if len(packet.payload) > 4 + 2 * packet.token_count:
            raise StreamError("payload exceeds maximum rANS symbol cost")
        try:
            ids = self.entropy.decode(packet.payload, packet.token_count, packet.block_id)
        except ValueError as error:
            raise StreamError(str(error)) from error
        self.token_ids.extend(ids)
        self.packets.append(wire)

    def byte_counts(self):
        packet_bytes = sum(map(len, self.packets))
        frame_bytes = len(self.header.to_bytes())
        overhead = len(self.packets) * PACKET_HEADER_BYTES
        return {"frame_header_bytes": frame_bytes, "packet_header_bytes": overhead,
                "payload_bytes": packet_bytes - overhead, "side_info_bytes": 0,
                "total_bytes": frame_bytes + packet_bytes,
                "transport_framing_bytes": 4 * (len(self.packets) + 1)}
