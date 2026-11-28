from io import BytesIO

from baselines.dcvc_uf.evaluate import account_stream
from third_party.DCVC.src.utils.stream_helper import write_ip, write_sps


def test_physical_rate_counts_intra_tail_and_actual_source_clock():
    stream = BytesIO()
    sps = write_sps(stream, {"sps_id": 0, "height": 1080, "width": 1920})
    i_bytes = write_ip(stream, True, 0, 16, 1, False, b"image")
    p_bytes = write_ip(stream, False, 0, 16, 1, False, b"inter")
    ledger = account_stream(stream.getvalue(), 4, 60, 8)

    assert ledger["physical_bytes"] == sps + i_bytes + p_bytes
    assert ledger["payload_bytes"] == 10
    assert ledger["i_bytes"] == i_bytes and ledger["p_bytes"] == p_bytes
    assert ledger["header_bytes"] == len(stream.getvalue()) - 10
    assert ledger["valid_frames"] == 4 and ledger["tail_padding_frames"] == 5
    assert ledger["bpp"] == len(stream.getvalue()) * 8 / (4 * 1080 * 1920)
    assert ledger["mbps"] == len(stream.getvalue()) * 8 / (4 / 60) / 1e6
