import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "aip_vtok.evaluation.gate0",
        "aip_vtok.training.smoke",
        "aip_vtok.training.formal",
        "aip_vtok.evaluation.formal",
        "aip_vtok.evaluation.report",
    ],
)
def test_cli_documents_arguments_without_opening_gpu_or_data(module, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [module, "--help"])
    with pytest.raises(SystemExit) as error:
        importlib.import_module(module).main()
    assert error.value.code == 0
    assert "--config" in capsys.readouterr().out
