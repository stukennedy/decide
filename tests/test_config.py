import pytest

from decide import config


@pytest.mark.parametrize("ram,env,bits", [
    (16, "", 8), (24, "", 8), (32, "", None), (128, "", None),
    (128, "8", 8), (16, "16", None), (16, "full", None), (64, "4", 4),
])
def test_model_bits(monkeypatch, ram, env, bits):
    monkeypatch.setattr(config, "_ram_gb", lambda: ram)
    monkeypatch.setenv("DECIDE_BITS", env)
    assert config.model_bits() == bits


def test_model_bits_rejects_nonsense(monkeypatch):
    monkeypatch.setenv("DECIDE_BITS", "3")
    with pytest.raises(ValueError):
        config.model_bits()
