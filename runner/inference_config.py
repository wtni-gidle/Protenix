"""Flat, optional YAML overrides for the prediction wrapper."""

from pathlib import Path

import click
import yaml


_CONFIG_TYPES = {
    "write_input_json": bool,
    "compress_fold_input": bool,
    "compress_full_confidence": bool,
    "skip": bool,
    "num_diffusion_samples": int,
    "num_diffusion_steps": int,
    "model_name": str,
    "dtype": str,
    "num_cycles": int,
    "use_msa": bool,
    "use_template": bool,
    "use_rna_msa": bool,
    "max_template_date": str,
}
_CONFIG_ARGUMENTS = {
    "num_cycles": "cycle",
    "num_diffusion_steps": "step",
    "num_diffusion_samples": "sample",
}


def load_config(config_path):
    """Return validated overrides; omitted keys keep their CLI/default values."""
    if config_path is None:
        return {}
    path = Path(config_path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise click.BadParameter(str(exc), param_hint="--config") from exc
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise click.BadParameter("Expected a flat YAML mapping.", param_hint="--config")
    overrides = {}
    for key, value in config.items():
        if key not in _CONFIG_TYPES:
            raise click.BadParameter(f"Unknown config key: {key!r}", param_hint="--config")
        expected = _CONFIG_TYPES[key]
        if type(value) is not expected:
            raise click.BadParameter(
                f"{key} must be {expected.__name__}, got {type(value).__name__}.",
                param_hint="--config",
            )
        overrides[_CONFIG_ARGUMENTS.get(key, key)] = value
    return overrides
