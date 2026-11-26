// The signed model uses Microsoft's DCVC-UF byte-rANS symbol ordering and CDFs.
// Independent PHT records require bounded decoding and a checked flush state.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <class T>
using Array = py::array_t<T, py::array::c_style>;
constexpr uint32_t scale_bits = 16;
constexpr uint32_t total = 1u << scale_bits;
constexpr uint32_t state_min = 1u << 23;
constexpr ssize_t symbol_limit = 4 * 1024 * 1024;

class Reader {
public:
    explicit Reader(const std::string& payload)
        : next(reinterpret_cast<const uint8_t*>(payload.data())), end(next + payload.size()) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            state |= static_cast<uint32_t>(byte()) << shift;
        }
        if (state < state_min || state >= state_min * 256) fail();
    }

    unsigned read(const int32_t* cdf, ssize_t count, const uint8_t* lookup = nullptr) {
        const auto slot = static_cast<int32_t>(state & (total - 1));
        const auto index = lookup ? lookup[slot] : std::upper_bound(cdf, cdf + count, slot) - cdf - 1;
        state = (cdf[index + 1] - cdf[index]) * (state >> scale_bits) + slot - cdf[index];
        renormalize();
        return index;
    }

    int8_t signed_symbol(const int32_t* cdf, ssize_t count, const uint8_t* lookup = nullptr) {
        unsigned index = read(cdf, count, lookup);
        if (index == count - 2) {
            unsigned n = bits();
            if (n == 3) n += bits();
            if (n > 4) fail();
            unsigned raw = 0;
            for (unsigned i = 0; i < n; ++i) raw |= bits() << (2 * i);
            index += raw;
        }
        if (index > 256) fail();
        return index % 2 ? static_cast<int8_t>((index + 1) / 2) : -static_cast<int>(index / 2);
    }

    void finish() const {
        if (state != state_min || next != end) fail();
    }

private:
    [[noreturn]] static void fail() {
        throw std::invalid_argument("invalid or truncated PHT rANS record");
    }
    uint8_t byte() {
        if (next == end) fail();
        return *next++;
    }
    void renormalize() {
        while (state < state_min) state = (state << 8) | byte();
    }
    unsigned bits() {
        unsigned value = state & 3u;
        state >>= 2;
        renormalize();
        return value;
    }

    const uint8_t* next;
    const uint8_t* end;
    uint32_t state = 0;
};

static void check_cdf(const int32_t* row, ssize_t length) {
    if (length < 3 || length > 257 || row[0] != 0 || row[length - 1] != total
        || std::adjacent_find(row, row + length, std::greater_equal<int32_t>()) != row + length) {
        throw std::invalid_argument("invalid PHT integer CDF");
    }
}

class SignedDecoder {
public:
    SignedDecoder(Array<int32_t> cdfs, Array<int32_t> sizes, bool dense_lookup) {
        if (cdfs.ndim() != 2 || sizes.ndim() != 1 || sizes.size() != cdfs.shape(0)
            || sizes.size() < 1 || sizes.size() > 16384) {
            throw std::invalid_argument("invalid signed rANS bank");
        }
        for (ssize_t i = 0; i < sizes.size(); ++i) {
            const auto length = *sizes.data(i);
            const auto* row = cdfs.data(i, 0);
            if (length > cdfs.shape(1)) throw std::invalid_argument("truncated rANS table");
            check_cdf(row, length);
            rows.emplace_back(row, row + length);
            if (dense_lookup) {
                auto& map = maps.emplace_back(total);
                for (int symbol = 0; symbol < length - 1; ++symbol) {
                    std::fill(map.begin() + row[symbol], map.begin() + row[symbol + 1], symbol);
                }
            }
        }
    }

    Array<int8_t> y(const std::string& raw, Array<uint8_t> indexes) const {
        if (indexes.ndim() != 1) throw std::invalid_argument("expected flat HT-S indexes");
        return decode(raw, indexes.size(), [&](ssize_t i) { return indexes.data()[i]; });
    }

    Array<int8_t> z(const std::string& raw, int count, unsigned offset, unsigned stride) const {
        if (!stride || offset >= rows.size() || stride > rows.size() - offset) {
            throw std::invalid_argument("invalid hyperprior CDF range");
        }
        return decode(raw, count, [&](ssize_t i) { return offset + i % stride; });
    }

private:
    template<class Select>
    Array<int8_t> decode(const std::string& raw, ssize_t count, Select index_of) const {
        if (count < 0 || count > symbol_limit) throw std::invalid_argument("invalid symbol count");
        Array<int8_t> output(count);
        auto* dst = output.mutable_data();
        {
            py::gil_scoped_release unlocked;
            Reader reader(raw);
            for (ssize_t i = 0; i < count; ++i) {
                const auto index = static_cast<std::size_t>(index_of(i));
                if (index >= rows.size()) throw std::invalid_argument("signed rANS index out of range");
                const auto& row = rows[index];
                dst[i] = reader.signed_symbol(row.data(), row.size(), maps.empty() ? nullptr : maps[index].data());
            }
            reader.finish();
        }
        return output;
    }

    std::vector<std::vector<int32_t>> rows;
    std::vector<std::vector<uint8_t>> maps;
};

class PrecisionCoder {
public:
    explicit PrecisionCoder(Array<int32_t> cdfs) {
        if (cdfs.ndim() != 2 || cdfs.shape(0) < 1 || cdfs.shape(1) < 3) {
            throw std::invalid_argument("invalid precision CDF bank");
        }
        columns = cdfs.shape(1);
        for (ssize_t i = 0; i < cdfs.shape(0); ++i) check_cdf(cdfs.data(i, 0), columns);
        tables.assign(cdfs.data(), cdfs.data() + cdfs.size());
    }

    py::bytes encode(Array<uint8_t> values, Array<uint16_t> indexes) const {
        validate(indexes);
        if (values.ndim() != 1 || values.size() != indexes.size()) {
            throw std::invalid_argument("invalid precision symbols");
        }
        std::vector<uint8_t> scratch(2 * values.size() + 4);
        auto* end = scratch.data() + scratch.size();
        auto* cursor = end;
        uint32_t state = state_min;
        {
            py::gil_scoped_release unlocked;
            for (ssize_t i = values.size(); i-- > 0;) {
                const auto value = values.data()[i];
                if (value >= columns - 1) throw std::invalid_argument("precision symbol out of range");
                const auto* cdf = row(indexes.data()[i]);
                const uint32_t frequency = cdf[value + 1] - cdf[value];
                const uint32_t maximum = frequency << (23 - scale_bits + 8);
                while (state >= maximum) {
                    *--cursor = static_cast<uint8_t>(state);
                    state >>= 8;
                }
                state = ((state / frequency) << scale_bits) + state % frequency + cdf[value];
            }
        }
        cursor -= 4;
        for (unsigned shift = 0; shift < 32; shift += 8) cursor[shift / 8] = state >> shift;
        return py::bytes(reinterpret_cast<const char*>(cursor), end - cursor);
    }

    Array<uint8_t> decode(const std::string& raw, Array<uint16_t> indexes) const {
        validate(indexes);
        Array<uint8_t> output(indexes.size());
        auto* dst = output.mutable_data();
        {
            py::gil_scoped_release unlocked;
            Reader reader(raw);
            for (ssize_t i = 0; i < indexes.size(); ++i) {
                dst[i] = reader.read(row(indexes.data()[i]), columns);
            }
            reader.finish();
        }
        return output;
    }

private:
    void validate(const Array<uint16_t>& indexes) const {
        if (indexes.ndim() != 1 || indexes.size() > symbol_limit) {
            throw std::invalid_argument("invalid precision indexes");
        }
    }
    const int32_t* row(uint16_t index) const {
        if (static_cast<std::size_t>(index) >= tables.size() / columns) {
            throw std::invalid_argument("precision rANS index out of range");
        }
        return tables.data() + columns * index;
    }

    ssize_t columns;
    std::vector<int32_t> tables;
};

PYBIND11_MODULE(pht_entropy_native, module) {
    py::class_<SignedDecoder>(module, "SignedDecoder")
        .def(py::init<Array<int32_t>, Array<int32_t>, bool>())
        .def("y", &SignedDecoder::y)
        .def("z", &SignedDecoder::z);
    py::class_<PrecisionCoder>(module, "PrecisionCoder")
        .def(py::init<Array<int32_t>>())
        .def("encode", &PrecisionCoder::encode)
        .def("decode", &PrecisionCoder::decode);
}
