from baselines.dcvc_uf.units import Unit
from experiments.pht_tok.network_dual import oracle_prefix
from pht_tok.streaming import RateTrace


def test_oracle_selects_the_independent_refresh_without_speculative_decode():
    refresh = Unit(0, 1, 1, True, False)
    assert oracle_prefix(None, None, None, refresh, RateTrace(((0, 1),)), 0, 0, 0, None) == 1
