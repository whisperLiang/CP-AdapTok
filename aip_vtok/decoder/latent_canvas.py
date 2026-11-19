"""New-token-only scatter with explicit arrival masks; ID 0 is a valid symbol."""

import torch


class LatentCanvas:
    def __init__(self, tokenizer, plan, prediction):
        self.tokenizer, self.plan, self.prediction = tokenizer, plan, prediction
        self.buffers = [torch.zeros_like(prediction) for _ in range(3)]
        self.masks = [
            torch.zeros(
                1,
                1,
                prediction.shape[-2] // size,
                prediction.shape[-1] // size,
                device=prediction.device,
                dtype=torch.bool,
            )
            for size in (4, 2, 1)
        ]
        self.count, self.projected_tokens = 0, 0

    def append_ids(self, ids, offset):
        self.append_codes(self.tokenizer.fsq.ids_to_codes(ids), offset)

    def append_codes(self, codes, offset):
        count = codes.shape[1]
        if offset != self.count or not 0 < count <= 128 or offset + count > len(self.plan.positions):
            raise ValueError("nonconsecutive, duplicate, missing, or oversized token block")
        passes = self.plan.pass_ids[offset : offset + count]
        positions = self.plan.positions[offset : offset + count]
        for pass_id in range(3):
            selected = passes == pass_id
            if not selected.any():
                continue
            spatial, values = self.tokenizer.lift_tokens(
                codes[:, selected], positions[selected], pass_id, self.plan.latent_shape
            )
            buffer = self.buffers[pass_id].flatten(2)
            indices = spatial[None, None].expand_as(values)
            if torch.is_grad_enabled():
                self.buffers[pass_id] = buffer.scatter_add(2, indices, values).reshape_as(self.prediction)
            else:
                buffer.scatter_add_(2, indices, values)
            self.masks[pass_id].flatten()[positions[selected]] = True
        self.count += count
        self.projected_tokens += count

    def latent(self):
        return self.prediction + self.buffers[0] + self.buffers[1] + self.buffers[2]
