import torch

from cp_adaptok.streaming.entropy_codec import BlockEntropyModel


def test_learning_and_export_reduce_actual_packet_bytes():
    model = BlockEntropyModel(codebook_size=32, blocks=2)
    ids = torch.zeros(1, 256, dtype=torch.long)
    initial = model.export()
    before = initial.encode(ids[0, :128].tolist(), 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    for _ in range(30):
        optimizer.zero_grad()
        model.loss(ids, 128).backward()
        optimizer.step()
    codec = model.export()
    after = codec.encode(ids[0, :128].tolist(), 0)
    assert len(after) < len(before) / 2
    assert codec.entropy_id != initial.entropy_id
    assert codec.decode(after, 128, 0) == [0] * 128
