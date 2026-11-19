"""Stage 1 model/optimizer construction shared by bounded and formal training."""

from dataclasses import dataclass

import torch

from aip_vtok.models.codec import create_codec
from aip_vtok.training.determinism import configure_determinism
from aip_vtok.training.losses import ReconstructionLoss
from aip_vtok.training.trainer import RNGStreams


@dataclass
class TrainingComponents:
    codec: torch.nn.Module
    objective: torch.nn.Module
    optimizer: torch.optim.Optimizer
    parameters: list
    rng: RNGStreams


def create_components(config):
    configure_determinism()
    codec = create_codec(config).train()
    objective = ReconstructionLoss(config["stage1"]["loss"]).cuda()
    parameters = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["stage1"]["lr"], weight_decay=0)
    return TrainingComponents(codec, objective, optimizer, parameters, RNGStreams(config["seed"]))
