"""Stage 1 model/optimizer construction shared by bounded and formal training."""

from dataclasses import dataclass

import torch

from aip_vtok.models.codec import MODEL_VERSION, create_codec
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


def create_components(config, *, model_version=MODEL_VERSION, joint_projector=False):
    configure_determinism()
    codec = create_codec(config, model_version=model_version).train()
    objective = ReconstructionLoss(config["stage1"]["loss"]).cuda()
    parameters = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["stage1"]["lr"], weight_decay=0)
    if joint_projector:
        enable_projector(codec, optimizer, parameters)
    return TrainingComponents(codec, objective, optimizer, parameters, RNGStreams(config["seed"]))


def enable_projector(codec, optimizer, parameters):
    group = list(codec.projector.requires_grad_(True).parameters())
    optimizer.add_param_group({"params": group, "lr": optimizer.param_groups[0]["lr"] * 0.1})
    parameters.extend(group)
