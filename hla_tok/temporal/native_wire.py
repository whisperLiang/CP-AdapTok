"""Optional host-side HLT Record packing and parsing in one native call per batch."""

import ctypes
from pathlib import Path

import numpy as np

from hla_tok.streaming.temporal_wire import HEADER_BYTES, RECORD_BYTES, VERSION, Mode
from hla_tok.temporal.profile import Profile

NativeLedger = tuple[int, int, int, int, int]

_ERRORS = {
    -1: "truncated HLT record header",
    -2: "invalid HLT record length",
    -3: "invalid HLT record CRC",
    -4: "duplicate or out-of-range HLT Record",
    -5: "overlapping or invalid HLT scan range",
    -6: "coding mode conflicts with the bound Profile",
    -7: "invalid HLT block payload or padding",
}


class NativeWire:
    """The profile owns legal indices and lengths. Native code checks each received Record."""

    def __init__(self, library: Path, profile: Profile):
        self.path = library.resolve()
        dll = ctypes.CDLL(str(self.path))
        dll.hlt_record_abi.argtypes = []
        dll.hlt_record_abi.restype = ctypes.c_uint32
        if dll.hlt_record_abi() != ((VERSION << 16) | RECORD_BYTES):
            raise ValueError("native HLT library has a different Record ABI")
        pointer, number, size = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        dll.write_hlt.argtypes = [pointer, pointer, pointer, number, number, number, number, number, pointer, size]
        dll.write_hlt.restype = number
        dll.open_hlt.argtypes = [ctypes.c_char_p, size, pointer, number, number, ctypes.c_uint8,
                                 pointer, pointer, pointer]
        dll.open_hlt.restype = number
        self._writer, self._reader = dll.write_hlt, dll.open_hlt
        self.profile = profile
        self._starts = np.asarray(profile.starts, dtype=np.int32)
        self._starts.setflags(write=False)
        self._modes = sum(1 << mode for mode in Mode if profile.mode_allowed(mode))

    def pack(self, codes: np.ndarray, modes: np.ndarray, lengths: np.ndarray, fit: int, width: int,
             first_record: int, first_slot: int, base: bool) -> bytes:
        limits = ((codes, np.int8, fit * width), (modes, np.uint8, fit), (lengths, np.int32, fit))
        if not 1 <= fit <= self.profile.records or not 1 <= width <= self.profile.block_size:
            raise ValueError("native HLT block dimensions do not fit the bound Profile")
        if any(array.dtype != dtype or array.ndim != 1 or not array.flags.c_contiguous or array.size < size
               for array, dtype, size in limits):
            raise ValueError("native HLT writer requires contiguous symbol, mode and length arrays")
        count = fit * RECORD_BYTES + int(lengths[:fit].sum())
        output = ctypes.create_string_buffer(count)
        emitted = self._writer(codes.ctypes.data, modes.ctypes.data, lengths.ctypes.data, fit, width,
                               first_record, first_slot, base, output, count)
        if emitted != count:
            raise RuntimeError("native HLT writer disagrees with the selected physical lengths")
        return output.raw

    def open(self, raw: memoryview, output: np.ndarray) -> tuple[set[int], NativeLedger]:
        if (output.dtype != np.int8 or output.shape != self.profile.scan.shape
                or not output.flags.c_contiguous or not output.flags.writeable):
            raise ValueError("native HLT reader requires a writable contiguous signed scan buffer")
        body = bytes(raw[HEADER_BYTES:])
        seen = np.zeros(self.profile.records, dtype=np.uint8)
        ledger = np.zeros(5, dtype=np.uint32)
        result = self._reader(body, len(body), self._starts.ctypes.data, self.profile.records,
                              self.profile.base_records, self._modes, output.ctypes.data,
                              seen.ctypes.data, ledger.ctypes.data)
        if result:
            raise ValueError(_ERRORS.get(result, "invalid native HLT Record"))
        count, payload, base, bitmap, padding = map(int, ledger)
        return set(np.flatnonzero(seen).tolist()), (count, payload, base, bitmap, padding)

    def matches(self, profile: Profile) -> bool:
        return profile.binding == self.profile.binding
