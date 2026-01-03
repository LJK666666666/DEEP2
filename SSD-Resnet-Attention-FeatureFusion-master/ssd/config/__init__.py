from .defaults import _C as cfg


def load_config_file(config_file):
    """
    Load config file with UTF-8 encoding support.
    This fixes the issue where Chinese comments in yaml files
    cause encoding errors on Windows (GBK vs UTF-8).
    """
    import yaml
    import io

    with io.open(config_file, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)

    if cfg_dict is not None:
        cfg.merge_from_other_cfg(cfg.clone())  # reset
        _merge_dict_to_cfg(cfg_dict, cfg)


def _merge_dict_to_cfg(cfg_dict, cfg_node):
    """Recursively merge dict into cfg node."""
    for k, v in cfg_dict.items():
        if hasattr(cfg_node, k):
            if isinstance(v, dict):
                _merge_dict_to_cfg(v, getattr(cfg_node, k))
            else:
                setattr(cfg_node, k, v)
