// UF enhancement segments use the upstream (0,+1,-1,...) rANS alphabet.
// Batched packing and reciprocal division follow the CC-S implementation.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <omp.h>
#include <tuple>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zlib.h>

#ifdef UFV2_PROFILE_CANVAS
#include <chrono>
static thread_local int64_t canvas_update_ns = 0;
#endif

namespace py = pybind11;
template<class T> using Array = py::array_t<T, py::array::c_style>;
constexpr uint32_t precision = 1 << 16;
constexpr uint32_t lower = 1 << 23;
constexpr uint8_t raw_flag = 128;
constexpr int header_bytes = 12;

struct Region {
    int c, y, x, channels, height, width;
    int count() const { return channels * height * width; }
};

class Layout {
public:
    Layout(int channels, int height, int width, int tile, int band)
        : channels(channels), height(height), width(width), stride(0) {
        if (channels < 1 || height < 1 || width < 1 || tile < 1 || band < 1
            || tile > 90 || int64_t(tile) * tile * band > 8192
            || int64_t(channels) * height * width > (1 << 24))
            throw std::invalid_argument("invalid enhancement layout");
        stride = tile * tile * band;
        for (int y = 0; y < height; y += tile)
            for (int x = 0; x < width; x += tile)
                for (int c = 0; c < channels; c += band)
                    regions.push_back({c, y, x, std::min(band, channels - c),
                                       std::min(tile, height - y), std::min(tile, width - x)});
    }

    template<class F> void rows(const Region& region, F visit) const {
        for (int c = region.c; c < region.c + region.channels; ++c) {
            for (int y = region.y; y < region.y + region.height; ++y)
                visit((c * height + y) * width + region.x, region.width);
        }
    }

    py::tuple gather(const Array<int8_t>& symbols) const {
        if (symbols.ndim() != 4 || symbols.shape(0) != 2 || symbols.shape(1) != channels
            || symbols.shape(2) != height || symbols.shape(3) != width)
            throw std::invalid_argument("symbols differ from group layout");
        const int groups = regions.size();
        Array<int8_t> values({2, groups, stride});
        Array<int32_t> statistics({2, groups, 2});
        const auto* source = symbols.data();
        auto* destination = values.mutable_data();
        auto* stats = statistics.mutable_data();
        const int plane = channels * height * width;
        {
            py::gil_scoped_release release;
            int invalid = 0;
            const int workers = plane >= (1 << 20) ? 8 : 4;
            #pragma omp parallel for if (plane >= 65536) num_threads(workers) reduction(|:invalid)
            for (int index = 0; index < 2 * groups; ++index) {
                const int level = index / groups;
                const Region& region = regions[index % groups];
                auto* target = destination + index * stride;
                int magnitude = 0, power = 0;
                rows(region, [&](int start, int length) {
                    for (int x = start; x < start + length; ++x) {
                        const int symbol = source[level * plane + x];
                        invalid |= symbol == -128;
                        *target++ = static_cast<int8_t>(symbol);
                        magnitude += std::abs(symbol);
                        power += symbol * symbol;
                    }
                });
                std::fill(target, destination + (index + 1) * stride, 0);
                stats[index * 2] = magnitude;
                stats[index * 2 + 1] = power;
            }
            if (invalid) throw std::invalid_argument("invalid signed symbol");
        }
        return py::make_tuple(std::move(values), std::move(statistics));
    }

    int channels, height, width, stride;
    std::vector<Region> regions;
};

static uint32_t get_le(const uint8_t* input, unsigned size) {
    uint32_t value = 0;
    for (unsigned i = 0; i < size; ++i) value |= uint32_t(input[i]) << (i * 8);
    return value;
}

static void put_le(uint8_t* output, uint32_t value, unsigned size) {
    for (unsigned i = 0; i < size; ++i) output[i] = static_cast<uint8_t>(value >> (i * 8));
}

struct Symbol {
    uint32_t start, frequency, threshold, reciprocal;
};

struct Table {
    std::vector<Symbol> symbols;
    std::vector<uint8_t> lookup;

    explicit Table(const std::vector<int>& cdf) : lookup(precision) {
        if (cdf.size() < 3 || cdf.size() > 129 || cdf.front() != 0 || cdf.back() != precision
            || std::adjacent_find(cdf.begin(), cdf.end(), std::greater_equal<int>()) != cdf.end())
            throw std::invalid_argument("noncanonical CDF");
        for (unsigned i = 0; i + 1 < cdf.size(); ++i) {
            const auto frequency = static_cast<uint32_t>(cdf[i + 1] - cdf[i]);
            symbols.push_back({static_cast<uint32_t>(cdf[i]), frequency, frequency << 15,
                               std::numeric_limits<uint32_t>::max() / frequency});
            std::fill(lookup.begin() + cdf[i], lookup.begin() + cdf[i + 1], i);
        }
    }
};

static void renormalize_write(uint32_t& state, uint8_t*& out, uint32_t threshold) {
    while (state >= threshold) {
        *--out = static_cast<uint8_t>(state);
        state >>= 8;
    }
}

static void put_bits(uint32_t& state, uint8_t*& out, unsigned value) {
    renormalize_write(state, out, 1u << 29);
    state = (state << 2) | value;
}

static void put_escape(uint32_t& state, uint8_t*& out, unsigned value) {
    int length = 0;
    while (value >> (2 * length)) ++length;
    for (int digit = length; digit > 0; --digit)
        put_bits(state, out, (value >> (2 * (digit - 1))) & 3);
    if (length >= 3) put_bits(state, out, length - 3);
    put_bits(state, out, std::min(3, length));
}

static uint8_t* encode(uint8_t* out, const int8_t* values, int count, const Table& table) {
    uint32_t state = lower;
    const int escape = table.symbols.size() - 1;
    while (count > 0) {
        const int input = values[--count];
        const int index = std::abs(input) * 2 - (input > 0);
        if (index >= escape) put_escape(state, out, index - escape);
        const Symbol& symbol = table.symbols[std::min(index, escape)];
        renormalize_write(state, out, symbol.threshold);
        const uint32_t quotient = (uint64_t(state) * symbol.reciprocal) >> 32;
        const uint32_t remainder = state - quotient * symbol.frequency;
        const uint32_t correction = remainder >= symbol.frequency;
        state = ((quotient + correction) << 16) + remainder - correction * symbol.frequency + symbol.start;
    }
    out -= 4;
    put_le(out, state, 4);
    return out;
}

class Decoder {
public:
    Decoder(std::string_view payload, const Table& table)
        : table(table), cursor(reinterpret_cast<const uint8_t*>(payload.data())), end(cursor + payload.size()) {
        for (unsigned shift = 0; shift < 32; shift += 8) state |= uint32_t(take()) << shift;
        if (state < lower || state >= (lower << 8)) throw std::invalid_argument("invalid initial rANS state");
    }

    int8_t read() {
        const uint32_t slot = state & (precision - 1);
        unsigned index = table.lookup[slot];
        const Symbol& symbol = table.symbols[index];
        state = symbol.frequency * (state >> 16) + slot - symbol.start;
        renormalize();
        if (index + 1 == table.symbols.size()) index += escape();
        if (index > 254) throw std::invalid_argument("unsigned or out-of-range symbol");
        return index % 2 ? int8_t((index + 1) / 2) : -int8_t(index / 2);
    }

    void finish() const {
        if (state != lower || cursor != end) throw std::invalid_argument("rANS terminal state mismatch");
    }

private:
    uint8_t take() {
        if (cursor == end) throw std::invalid_argument("truncated rANS segment");
        return *cursor++;
    }
    void renormalize() {
        while (state < lower) state = (state << 8) | take();
    }
    unsigned bits() {
        const unsigned value = state & 3;
        state >>= 2;
        renormalize();
        return value;
    }
    unsigned escape() {
        unsigned length = bits();
        if (length == 3) length += bits();
        if (length > 4) throw std::invalid_argument("invalid rANS escape length");
        unsigned value = 0;
        for (unsigned digit = 0; digit < length; ++digit) value |= bits() << (2 * digit);
        return value;
    }

    const Table& table;
    const uint8_t* cursor;
    const uint8_t* end;
    uint32_t state = 0;
};

static int compact_size(int count) { return (count + 1) / 2; }

static void write_raw4(uint8_t* out, const int8_t* values, int count) {
    std::fill(out, out + compact_size(count), 0);
    for (int i = 0; i < count; ++i)
        out[i / 2] |= static_cast<uint8_t>((values[i] + 8) << (4 * (i % 2)));
}

static void read_raw(std::string_view payload, int8_t* values, int count) {
    const auto* input = reinterpret_cast<const uint8_t*>(payload.data());
    if (count >= 8 && static_cast<int>(payload.size()) == compact_size(count)) {
        if (count % 2 && input[count / 2] >= 16)
            throw std::invalid_argument("noncanonical Raw4 padding");
        for (int i = 0; i < count; ++i)
            values[i] = ((input[i / 2] >> (4 * (i % 2))) & 15) - 8;
        return;
    }
    if (static_cast<int>(payload.size()) != count) throw std::invalid_argument("invalid raw segment length");
    std::copy_n(input, count, values);
    if (std::find(values, values + count, -128) != values + count)
        throw std::invalid_argument("invalid raw signed alphabet");
}

using Entry = std::tuple<double, int, int>;

static std::vector<Entry> order_records(const double* scores, const uint8_t* active, int groups, bool ranked) {
    std::vector<Entry> result;
    result.reserve(2 * groups);
    if (!ranked) {
        for (int g = 0; g < groups; ++g)
            if (active[g] || active[groups + g]) result.emplace_back(0, 0, g);
        for (int g = 0; g < groups; ++g)
            if (active[groups + g]) result.emplace_back(0, 1, g);
        return result;
    }
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    for (int g = 0; g < groups; ++g)
        if (active[g] || active[groups + g]) queue.emplace(-scores[g], 0, g);
    while (!queue.empty()) {
        auto entry = queue.top();
        queue.pop();
        result.push_back(entry);
        const auto [score, level, group] = entry;
        if (level == 0 && active[groups + group])
            queue.emplace(-scores[groups + group], 1, group);
    }
    return result;
}

struct Packet {
    int group, level, mode;
    std::string_view payload;
};

class Records {
public:
    explicit Records(const std::vector<std::vector<int>>& cdfs) {
        if (cdfs.empty() || cdfs.size() > raw_flag) throw std::invalid_argument("invalid CDF bank size");
        for (const auto& cdf : cdfs) tables.emplace_back(cdf);
    }

    py::tuple pack(const Layout& layout, const Array<int8_t>& values, const Array<double>& scores,
                   const Array<uint8_t>& active, const Array<uint8_t>& cdfs, bool ranked,
                   int cap, bool fallback) const {
        const int groups = layout.regions.size();
        if (values.ndim() != 3 || values.shape(0) != 2 || values.shape(1) != groups
            || values.shape(2) != layout.stride || scores.size() != 2 * groups
            || active.size() != 2 * groups || cdfs.size() != 2 * groups || cap < 0 || cap > (16 << 20))
            throw std::invalid_argument("invalid enhancement batch");
        if (!std::all_of(scores.data(), scores.data() + scores.size(), [](double s) { return std::isfinite(s); })
            || *std::max_element(cdfs.data(), cdfs.data() + cdfs.size()) >= tables.size())
            throw std::invalid_argument("invalid priority or CDF index");
        std::string bytes;
        bytes.reserve(cap);
        std::vector<int> ends;
        {
            py::gil_scoped_release release;
            const auto order = order_records(scores.data(), active.data(), groups, ranked);
            pack_batches(layout, values.data(), cdfs.data(), order, cap, fallback, bytes, ends);
        }
        return py::make_tuple(py::bytes(bytes), ends);
    }

    std::tuple<int, int, int> receive(const Layout& layout, const py::buffer& input, Array<int16_t>& canvas,
                                     Array<uint8_t>& levels, int bands, bool raw) const {
        check_canvas(layout, canvas);
        py::buffer_info buffer = input.request();
        if (buffer.ndim != 1 || buffer.itemsize != 1 || buffer.strides[0] != 1
            || levels.size() != static_cast<ssize_t>(layout.regions.size()))
            throw std::invalid_argument("invalid enhancement receiver buffer");
        auto* data = static_cast<const uint8_t*>(buffer.ptr);
        auto* output = canvas.mutable_data();
        auto* seen = levels.mutable_data();
        py::gil_scoped_release release;
        std::vector<Packet> packets;
        int offset = 0;
        while (buffer.size - offset >= header_bytes) {
            const auto* packet = data + offset;
            const uint32_t group = get_le(packet, 4);
            const int level = packet[4], mode = packet[5], size = get_le(packet + 6, 2);
            if (size < 4 || group >= layout.regions.size() || level > 1)
                throw std::invalid_argument("invalid enhancement header");
            if (buffer.size - offset < header_bytes + size) break;
            const uint8_t* payload = packet + header_bytes;
            if (get_le(packet + 8, 4) != crc32(crc32(0, packet, 8), payload, size))
                throw std::invalid_argument("record checksum mismatch");
            validate_index(group, level, mode, bands, raw, seen);
            packets.push_back({static_cast<int>(group), level, mode,
                               {reinterpret_cast<const char*>(payload), static_cast<size_t>(size)}});
            offset += header_bytes + size;
        }
        const int symbols = restore_packets(layout, packets, output);
        return {offset, packets.size(), symbols};
    }

private:
    static void check_canvas(const Layout& layout, const Array<int16_t>& canvas) {
        if (canvas.ndim() != 3 || canvas.shape(0) != layout.channels || canvas.shape(1) != layout.height
            || canvas.shape(2) != layout.width) throw std::invalid_argument("canvas differs from latent layout");
    }

    void validate_index(int group, int level, int mode, int bands, bool raw, uint8_t* seen) const {
        const int index = mode & (raw_flag - 1);
        const bool right_level = bands ? index == level * bands + group % bands : index / 4 == level;
        if (!right_level || index >= static_cast<int>(tables.size()) || ((mode & raw_flag) && !raw))
            throw std::invalid_argument("CDF does not match enhancement record");
        if (seen[group] > level) throw std::invalid_argument("duplicate enhancement");
        if (seen[group] < level) throw std::invalid_argument("missing refinement dependency");
        ++seen[group];
    }

    int restore_packets(const Layout& layout, const std::vector<Packet>& packets, int16_t* target) const {
        if (packets.empty()) return 0;
        int symbols = 0, invalid = 0;
        // Group ownership preserves coarse/fine ordering without shared writes.
        const int requested = packets.size() >= 1024 ? 8 : 4;
        #pragma omp parallel if(packets.size() >= 128) num_threads(requested) reduction(+:symbols) reduction(|:invalid)
        {
            const auto worker = static_cast<unsigned>(omp_get_thread_num());
            const auto workers = static_cast<unsigned>(omp_get_num_threads());
            std::vector<int8_t> values(layout.stride);
            try {
                for (const auto& packet : packets) {
                    if (static_cast<unsigned>(packet.group) % workers == worker)
                        symbols += restore_one(layout, packet, values.data(), target);
                }
            } catch (const std::invalid_argument&) {
                invalid = 1;
            }
        }
        if (invalid) throw std::invalid_argument("invalid enhancement entropy payload");
        return symbols;
    }

    int restore_one(const Layout& layout, const Packet& packet, int8_t* values, int16_t* target) const {
        const auto& region = layout.regions[packet.group];
        const int count = region.count();
        if (packet.mode & raw_flag) {
            read_raw(packet.payload, values, count);
        } else {
            Decoder decoder(packet.payload, tables[packet.mode]);
            for (int i = 0; i < count; ++i) values[i] = decoder.read();
            decoder.finish();
        }
        const int multiplier = 2 >> packet.level;
        int offset = 0;
#ifdef UFV2_PROFILE_CANVAS
        const auto update_start = std::chrono::steady_clock::now();
#endif
        layout.rows(region, [&](int start, int length) {
            for (int x = start; x < start + length; ++x)
                target[x] += values[offset++] * multiplier;
        });
#ifdef UFV2_PROFILE_CANVAS
        canvas_update_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - update_start).count();
#endif
        return count;
    }

    void pack_batches(const Layout& layout, const int8_t* values, const uint8_t* cdfs,
                      const std::vector<Entry>& order, int cap, bool fallback,
                      std::string& bytes, std::vector<int>& ends) const {
        const int groups = layout.regions.size();
        const int scratch_size = 4 * layout.stride + 4;
        constexpr int batch = 512;
        struct Slot {
            const uint8_t* payload;
            int size, mode;
        };
        std::array<Slot, batch> slots;
        // Each calling thread owns one workspace. OpenMP workers only write
        // into disjoint record slots through the caller's shared pointer.
        thread_local std::vector<uint8_t> workspace;
        workspace.resize(batch * scratch_size);
        auto* storage = workspace.data();
        const int records = order.size();
        for (int start = 0; start < records; start += batch) {
            const int count = std::min(batch, records - start);
            const int workers = count * layout.stride >= 262144 ? 8 : 4;
            #pragma omp parallel for if (count * layout.stride >= 32768) num_threads(workers) schedule(static)
            for (int i = 0; i < count; ++i) {
                const auto& [score, level, group] = order[start + i];
                const int index = level * groups + group;
                const int8_t* input = values + index * layout.stride;
                const int width = layout.regions[group].count();
                auto* tail = storage + (i + 1) * scratch_size;
                auto* begin = encode(tail, input, width, tables[cdfs[index]]);
                const auto [size, mode] = choose_payload(input, width, begin, tail, cdfs[index], fallback);
                slots[i] = {begin, size, mode};
            }
            for (int i = 0; i < count; ++i) {
                const auto& packet = slots[i];
                if (bytes.size() + header_bytes + packet.size > static_cast<unsigned>(cap)) return;
                const auto& [score, level, group] = order[start + i];
                append(bytes, group, level, packet.mode, packet.payload, packet.size);
                ends.push_back(bytes.size());
            }
        }
    }

    static std::pair<int, int> choose_payload(const int8_t* values, int count, uint8_t*& begin,
                                              uint8_t* end, int cdf, bool fallback) {
        const int encoded_size = end - begin;
        if (!fallback) return {encoded_size, cdf};
        if (count >= 8 && compact_size(count) < encoded_size
            && std::all_of(values, values + count, [](int v) { return v >= -8 && v <= 7; })) {
            write_raw4(begin, values, count);
            return {compact_size(count), cdf | raw_flag};
        }
        if (count >= 4 && count < encoded_size) {
            std::copy_n(values, count, begin);
            return {count, cdf | raw_flag};
        }
        return {encoded_size, cdf};
    }

    static void append(std::string& output, int group, int level, int mode, const uint8_t* data, int size) {
        std::array<uint8_t, header_bytes> header{};
        put_le(header.data(), group, 4);
        header[4] = static_cast<uint8_t>(level);
        header[5] = static_cast<uint8_t>(mode);
        put_le(header.data() + 6, size, 2);
        const auto crc = crc32(crc32(0, header.data(), 8), data, size);
        put_le(header.data() + 8, crc, 4);
        output.append(reinterpret_cast<const char*>(header.data()), header.size());
        output.append(reinterpret_cast<const char*>(data), size);
    }

    std::vector<Table> tables;
};

PYBIND11_MODULE(hla_uf_v2_records, module) {
#ifdef UFV2_PROFILE_CANVAS
    module.def("take_canvas_update_ns", []() {
        const auto elapsed = canvas_update_ns;
        canvas_update_ns = 0;
        return elapsed;
    });
#endif
    py::class_<Layout>(module, "Layout")
        .def(py::init<int, int, int, int, int>())
        .def("gather", &Layout::gather)
        .def_property_readonly("counts", [](const Layout& plan) {
            Array<int32_t> out(plan.regions.size());
            std::transform(plan.regions.begin(), plan.regions.end(), out.mutable_data(),
                           [](const Region& region) { return region.count(); });
            return out;
        });
    py::class_<Records>(module, "Records")
        .def(py::init<const std::vector<std::vector<int>>&>())
        .def("pack", &Records::pack)
        .def("receive", &Records::receive);
}
