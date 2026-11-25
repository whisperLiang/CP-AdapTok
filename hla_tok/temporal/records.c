#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <zlib.h>

enum { RAW4, RAW8, ZERO, BITMAP4 };
enum { RECORD_HEADER = 11, RECORD_OVERHEAD = 15, BASE_FLAG = 128 };

static uint32_t get16(const uint8_t *a) {
    return (uint32_t)a[0] | (uint32_t)a[1] << 8;
}
static uint32_t get32(const uint8_t *a) {
    return get16(a) | get16(a + 2) << 16;
}
static void put16(uint8_t *a, uint32_t v) {
    a[0] = (uint8_t)v;
    a[1] = (uint8_t)(v >> 8);
}
static void put32(uint8_t *a, uint32_t v) {
    put16(a, v);
    put16(a + 2, v >> 16);
}
static int8_t signed4(uint8_t word) {
    return (int8_t)(((word & 15) ^ 8) - 8);
}

static int write_payload(uint8_t *out, const int8_t *values, int width, int mode, int size) {
    if (mode == ZERO) return size == 0;
    if (mode == RAW8) {
        if (size != width) return 0;
        memcpy(out, values, (size_t)width);
        return 1;
    }
    if (mode == RAW4) {
        if (size != (width + 1) / 2) return 0;
        for (int col = 0; col < width; col += 2) {
            uint8_t other = col + 1 < width ? ((uint8_t)values[col + 1] & 15) : 0;
            out[col / 2] = ((uint8_t)values[col] & 15) | (uint8_t)(other << 4);
        }
        return 1;
    }
    int map = (width + 7) / 8;
    if (mode != BITMAP4 || size <= map) return 0;
    memset(out, 0, (size_t)size);
    int used = 0;
    for (int col = 0; col < width; col++) {
        if (values[col] == 0) continue;
        int pos = map + used / 2;
        if (pos >= size) return 0;
        out[col / 8] |= (uint8_t)(1u << (col % 8));
        out[pos] |= (uint8_t)(((uint8_t)values[col] & 15) << (4 * (used % 2)));
        used++;
    }
    return used != 0 && size == map + (used + 1) / 2;
}

int write_hlt(const int8_t *input, const uint8_t *modes, const int32_t *sizes, int rows, int width,
              int index, int scan, int base, uint8_t *wire, size_t capacity) {
    uint8_t *pos = wire;
    if (width < 1 || width > 256 || rows < 0) return -1;
    for (int row = 0; row < rows; row++) {
        int size = sizes[row];
        if (size < 0 || size > width || capacity < (size_t)(RECORD_OVERHEAD + size)) return -1;
        int body = RECORD_HEADER + size;
        put16(pos, (uint32_t)(index + row));
        put32(pos + 2, (uint32_t)(scan + row * width));
        pos[6] = modes[row] | (base ? BASE_FLAG : 0);
        put16(pos + 7, (uint32_t)width);
        put16(pos + 9, (uint32_t)size);
        if (!write_payload(pos + RECORD_HEADER, input + row * width, width, modes[row], size)) return -1;
        put32(pos + body, (uint32_t)crc32(0, pos, (uInt)body));
        pos += RECORD_OVERHEAD + size;
        capacity -= (size_t)(RECORD_OVERHEAD + size);
    }
    return (int)(pos - wire);
}

static int open_bitmap(int8_t *out, const uint8_t *payload, uint32_t count, uint32_t size) {
    uint32_t map = (count + 7) / 8;
    if (size <= map || (count % 8 && payload[map - 1] >> (count % 8))) return -1;
    uint32_t used = 0;
    for (uint32_t i = 0; i < map; i++) used += (uint32_t)__builtin_popcount(payload[i]);
    if (!used || size != map + (used + 1) / 2) return -1;
    if (used % 2 && payload[size - 1] >> 4) return -1;
    const uint8_t *words = payload + map;
    memset(out, 0, count);
    uint32_t symbol = 0;
    for (uint32_t col = 0; col < count; col++) {
        if (payload[col / 8] & (1u << (col % 8))) {
            out[col] = signed4((uint8_t)(words[symbol / 2] >> (4 * (symbol % 2))));
            if (!out[col]) return -1;
            symbol++;
        }
    }
    return (int)((8 - count % 8) % 8 + 4 * (used % 2));
}

static int open_payload(int8_t *out, const uint8_t *payload, uint32_t count, uint32_t size, uint8_t mode) {
    if (mode == BITMAP4) return open_bitmap(out, payload, count, size);
    if (mode == RAW8 && size == count) {
        memcpy(out, payload, count);
        return 0;
    }
    if (mode == ZERO && !size) {
        memset(out, 0, count);
        return 0;
    }
    if (mode != RAW4 || size != (count + 1) / 2 || (count % 2 && payload[size - 1] >> 4)) return -1;
    for (uint32_t i = 0; i < count; i++) out[i] = signed4((uint8_t)(payload[i / 2] >> (4 * (i % 2))));
    return (int)(4 * (count % 2));
}

int open_hlt(const uint8_t *wire, size_t size, const int32_t *starts, int records, int base,
             uint8_t allowed, int8_t *values, uint8_t *seen, uint32_t *ledger) {
    static const uint8_t allowed_max = BITMAP4;
    while (size > 0) {
        if (size < RECORD_OVERHEAD) return -1;
        uint32_t idx = get16(wire);
        uint32_t from = get32(wire + 2);
        uint8_t mode = wire[6] & 127;
        int is_base = (wire[6] & BASE_FLAG) != 0;
        uint32_t count = get16(wire + 7);
        uint32_t bytes = get16(wire + 9);
        if (count < 1 || count > 256 || bytes > count || size < RECORD_OVERHEAD + bytes) return -2;
        uint32_t body = RECORD_HEADER + bytes;
        if ((uint32_t)crc32(0, wire, body) != get32(wire + body)) return -3;
        if (idx >= (uint32_t)records || seen[idx]) return -4;
        if (from != (uint32_t)starts[idx] || count != (uint32_t)(starts[idx + 1] - starts[idx])
                || is_base != (idx < (uint32_t)base)) return -5;
        if (mode > allowed_max || !(allowed & (1u << mode))) return -6;
        int padding = open_payload(values + from, wire + RECORD_HEADER, count, bytes, mode);
        if (padding < 0) return -7;
        seen[idx] = 1;
        ledger[0]++;
        ledger[1] += bytes;
        if (is_base) ledger[2] += bytes;
        if (mode == BITMAP4) ledger[3] += (count + 7) / 8;
        ledger[4] += (uint32_t)padding;
        wire += RECORD_OVERHEAD + bytes;
        size -= RECORD_OVERHEAD + bytes;
    }
    return 0;
}

/* The Python adapter owns frame validation and the scan schedule. */
uint32_t hlt_record_abi(void) {
    return (1u << 16) | RECORD_OVERHEAD;
}
