#include "rans.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using Table = std::vector<int32_t>;
using Lookup = std::vector<uint8_t>;
constexpr uint32_t scale_bits = 16;
constexpr uint32_t precision = 1u << scale_bits;
constexpr uint32_t state_min = 1u << 23;
constexpr int segment_limit = 65531;

class DecoderState {
public:
    explicit DecoderState(const std::string& bytes)
        : cursor(reinterpret_cast<const uint8_t*>(bytes.data())), end(cursor + bytes.size()) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            state |= static_cast<uint32_t>(take()) << shift;
        }
        if (state < state_min || state >= (state_min << 8)) {
            throw std::invalid_argument("noncanonical initial rANS state");
        }
    }

    int8_t read(const Table& cdf, const Lookup& lookup) {
        auto frequency = static_cast<int32_t>(state & (precision - 1));
        unsigned value = lookup[frequency];
        state = (cdf[value + 1] - cdf[value]) * (state >> scale_bits) + frequency - cdf[value];
        renormalize();
        if (value == cdf.size() - 2) value += escape();
        if (value > 254) throw std::invalid_argument("rANS symbol outside signed alphabet");
        return value % 2 ? static_cast<int8_t>((value + 1) / 2) : -static_cast<int8_t>(value / 2);
    }

    void finish() const {
        if (state != state_min || cursor != end) throw std::invalid_argument("rANS terminal state mismatch");
    }

private:
    uint8_t take() {
        if (cursor == end) throw std::invalid_argument("truncated rANS segment");
        return *cursor++;
    }

    void renormalize() {
        while (state < state_min) state = (state << 8) | take();
    }

    unsigned bypass() {
        unsigned value = state & 3;
        state >>= 2;
        renormalize();
        return value;
    }

    unsigned escape() {
        unsigned length = bypass();
        if (length == 3) length += bypass();
        if (length > 4) throw std::invalid_argument("invalid rANS escape length");
        unsigned value = 0;
        for (unsigned digit = 0; digit < length; ++digit) value |= bypass() << (2 * digit);
        return value;
    }

    const uint8_t* cursor;
    const uint8_t* end;
    uint32_t state{0};
};

class SegmentCodec {
public:
    explicit SegmentCodec(std::vector<Table> input) : cdfs(std::move(input)) {
        if (cdfs.empty() || cdfs.size() > 256) throw std::invalid_argument("invalid CDF count");
        auto symbols = std::make_shared<std::vector<std::vector<RansSymbol>>>();
        auto max_values = std::make_shared<std::vector<int8_t>>();
        for (const auto& cdf : cdfs) {
            validate(cdf);
            symbols->emplace_back();
            auto& lookup = lookups.emplace_back(precision);
            for (std::size_t index = 0; index + 1 < cdf.size(); ++index) {
                symbols->back().push_back({static_cast<uint16_t>(cdf[index]),
                                          static_cast<uint16_t>(cdf[index + 1] - cdf[index])});
                std::fill(lookup.begin() + cdf[index], lookup.begin() + cdf[index + 1], index);
            }
            max_values->push_back(static_cast<int8_t>(cdf.size() - 2));
        }
        encoder.set_cdf(symbols, max_values, 1);
    }

    py::bytes encode(const py::array_t<int8_t, py::array::c_style>& values, int index) {
        check(index, values.size());
        if (values.ndim() != 1) throw std::invalid_argument("symbols must be one-dimensional");
        auto packed = std::make_shared<std::vector<int16_t>>(values.size());
        const auto* input = values.data();
        for (ssize_t position = 0; position < values.size(); ++position) {
            if (input[position] == -128) throw std::invalid_argument("invalid signed symbol");
            (*packed)[position] = static_cast<int16_t>(input[position] * 256 + index);
        }
        encoder.reset();
        encoder.encode_y_internal(packed, values.size(), 0);
        encoder.flush();
        const auto coded = encoder.get_encoded_stream();
        return py::bytes(reinterpret_cast<const char*>(coded->data()), coded->size());
    }

    py::array_t<int8_t> decode(const std::string& payload, int index, int count) const {
        check(index, count);
        DecoderState decoder(payload);
        py::array_t<int8_t> symbols({count}, {sizeof(int8_t)});
        auto* output = symbols.mutable_data();
        for (int position = 0; position < count; ++position) {
            output[position] = decoder.read(cdfs[index], lookups[index]);
        }
        decoder.finish();
        return symbols;
    }

private:
    static void validate(const Table& cdf) {
        if (cdf.size() < 3 || cdf.size() > 129 || cdf.front() != 0 || cdf.back() != precision
            || std::adjacent_find(cdf.begin(), cdf.end(), std::greater_equal<int32_t>()) != cdf.end()) {
            throw std::invalid_argument("noncanonical integer rANS CDF");
        }
    }

    void check(int index, ssize_t count) const {
        if (index < 0 || static_cast<std::size_t>(index) >= cdfs.size() || count < 1 || count > segment_limit) {
            throw std::invalid_argument("invalid rANS index or symbol count");
        }
    }

    std::vector<Table> cdfs;
    std::vector<Lookup> lookups;
    RansEncoderLib encoder;
};

PYBIND11_MODULE(hla_uf_rans, module) {
    py::class_<SegmentCodec>(module, "SegmentCodec")
        .def(py::init<std::vector<Table>>())
        .def("encode", &SegmentCodec::encode)
        .def("decode", &SegmentCodec::decode);
}
