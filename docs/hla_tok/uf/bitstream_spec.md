# HUF1 application packet

An access unit belongs to one RGB picture. The transport or file index gives its byte extent; HUF1 does not prescribe a packetization or network overhead model. The common header never encodes the requested bandwidth, final byte count or record count. A separately provided model/CDF bank and the pinned native checkpoints are prerequisites.

| Byte offset | Field | Encoding |
| ---: | --- | --- |
| 0 | Magic | 4 bytes, `HUF1` |
| 4 | Enhancement model/CDF tag | 8 bytes |
| 12 | Picture index | `uint32` |
| 16 | Original height, width | two `uint16` |
| 20 | Native QP | `uint8` |
| 21 | Flags | bit 0 I; bit 1 memory reset; bits 2–4 rANS parallel count minus one; remaining bits zero |
| 22 | Native payload length | `uint32` |
| 26 | CRC-32 | 4 bytes over header and native payload |
| 30 | Base payload | opaque Microsoft compressed I/P access unit, including jointly coded `z` and `y` |

All integers use little-endian ordering. Each optional group record has a 4-byte group ID, an 8-bit level (0 or 1), an 8-bit CDF identifier, a 2-byte payload length, a 4-byte CRC-32, then one independently terminated native-compatible 16-bit rANS segment. Group geometry is derived from the model tag's agreed tile/channel layout and the parsed dimensions. Boundary tiles send actual symbols only; the bitstream does not carry padded symbols. Signed escape codes and exact rANS terminal state are verified.

The encoder constructs the largest complete prefix once. The sender clips it at the last record boundary within `(20,833, 41,666, 83,333, 166,666)` B at nominal 30 FPS. An incomplete base is not a usable prefix. The separate 9,212 B base gate includes the 30-byte HUF1 header; a real overrun is recorded, not hidden by padding or a GOP mean. CRCs detect accidental corruption, not authenticity. No adversarial native bitstream or production network security guarantee is implied.

The receiver checks the native base once and commits its reference before applying any display record. It retains only the undecoded suffix and the accumulated integer canvas while growing a prefix. A partial record cannot affect the canvas. The complete coarse record is required before its fine record; the next P frame must follow a valid earlier base. Only an independent I picture can recover a base gap. Rendering an enhanced image can run again after receiving additional segments; it does not run the entropy decoder again for previous segments. Intentional enhancement drop is represented by a shorter legal prefix.
