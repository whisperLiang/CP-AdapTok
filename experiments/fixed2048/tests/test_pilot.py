from experiments.fixed2048.pilot import passed


def test_fitting_gate_uses_evaluated_loss_instead_of_configured_steps():
    history = {'step0_train':{'summary':{'global_mse':0.04}},
               'last_train_eval':{'summary':{'global_mse':0.02}}}
    assert passed(history, 0.05)
    history['last_train_eval']['summary']['global_mse'] = 0.041
    assert not passed(history, 0.05)
