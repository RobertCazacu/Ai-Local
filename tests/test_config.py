from config_utils import load_config


def test_load_config_has_defaults():
    cfg = load_config('nonexistent_config.json')
    assert 'store_dir' in cfg
    assert 'AUTO_ACCEPT_CONF' in cfg
