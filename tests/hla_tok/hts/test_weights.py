from base64 import b64encode

from scripts.hla_tok.hts.download_weights import file_hashes


def test_quickxor_wraps_and_folds_length(tmp_path):
    data = bytes(range(256)) * 3
    file = tmp_path / "weights"
    file.write_bytes(data)
    expected = bytearray(20)
    for i, value in enumerate(data):
        for bit in range(8):
            index = (i * 11 + bit) % 160
            expected[index // 8] ^= ((value >> bit) & 1) << (index % 8)
    for i, value in enumerate(len(data).to_bytes(8, "little"), 12):
        expected[i] ^= value
    assert file_hashes(file)["quickxor"] == b64encode(expected).decode()
