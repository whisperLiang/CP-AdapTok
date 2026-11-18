import copy

import pytest
import torch


def test_named_checkpoint_fork_copies_adam_and_common_weights_without_aliasing(small_model):
    from adaptok_hr.tests.test_training import settings
    from adaptok_hr.training.train import JointTrainer
    from experiments.adaptok_hr.fine_detail_v2.fork import adopt_training_state

    config = settings()
    trainer = JointTrainer(small_model, config)
    clip = torch.rand(2, 3, 128, 256)
    trainer.update(clip, 1)
    parent = {"model": copy.deepcopy(small_model.state_dict()), "optimizer": trainer.optimizer.state_dict(),
              "scheduler": trainer.scheduler.state_dict()}
    child_model = copy.deepcopy(small_model)
    # A new root parameter changes numeric Adam IDs of all existing parameters.
    child_model.register_parameter("fine_test_parameter", torch.nn.Parameter(torch.ones(1)))
    child = JointTrainer(child_model, config)
    with pytest.raises(ValueError, match="unapproved"):
        adopt_training_state(small_model, parent, child, allowed_new=())
    proof = adopt_training_state(small_model, parent, child, allowed_new=("fine_test_parameter",))
    assert proof["parent_optimizer_tensors"] == proof["child_inherited_optimizer_tensors"]
    assert proof["additional_parameters"] == ["fine_test_parameter"]
    assert child_model.fine_test_parameter not in child.optimizer.state
    for name, p in small_model.named_parameters():
        q = dict(child_model.named_parameters())[name]
        for key, value in trainer.optimizer.state[p].items():
            torch.testing.assert_close(child.optimizer.state[q][key], value, atol=0, rtol=0)
            assert value.data_ptr() != child.optimizer.state[q][key].data_ptr()
    state = torch.get_rng_state()
    trainer.update(clip, 2)
    torch.set_rng_state(state)
    child.update(clip, 2)
    for name, p in small_model.named_parameters():
        torch.testing.assert_close(p, dict(child_model.named_parameters())[name], rtol=0, atol=0)
