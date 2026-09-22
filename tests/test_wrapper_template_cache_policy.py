"""Reject persistent parsed-template caches at wrapper boundaries only."""
import json

import pytest
from ml_collections import ConfigDict

from protenix.data.inference.infer_dataloader import InferenceDataset
from protenix.data.template.template_finalizer import finalize_template_hits
from protenix.data.template.template_utils import TemplateHitFeaturizer
from runner import batch_inference
from runner.template_search import update_template_info


def configure(tmp_path, monkeypatch, cache):
    release = tmp_path / "release.json"
    release.write_text("{}")
    obsolete = tmp_path / "obsolete.json"
    obsolete.write_text("{}")
    values = dict(prot_template_cache_dir=cache, prot_template_mmcif_dir=str(tmp_path),
                  release_dates_path=str(release), obsolete_pdbs_path=str(obsolete),
                  kalign_binary_path="/bin/true", fetch_remote=False)
    for key, value in values.items():
        monkeypatch.setitem(batch_inference.data_configs["template"], key, value)


def dataset_config(tmp_path, cache):
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{"name": "job", "sequences": []}]))
    return ConfigDict(dict(input_json_path=str(source), dump_dir=str(tmp_path / "out"),
        use_msa=False, use_template=True, esm=dict(enable=False),
        data=dict(template=dict(prot_template_cache_dir=cache))))


def test_factory_rejects_persistent_cache(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, str(tmp_path / "old_cache"))
    with pytest.raises(ValueError, match="prot_template_cache_dir"):
        batch_inference._create_template_finalizer_featurizer(None, "2021-09-30")


def test_data_rejects_cache_even_when_templates_are_explicit(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, str(tmp_path / "old_cache"))
    task = [{"name": "job", "sequences": [{"proteinChain": {
        "sequence": "AAAAAA", "count": 1, "templates": []}}]}]
    with pytest.raises(ValueError, match="prot_template_cache_dir"):
        update_template_info(task, out_dir=str(tmp_path))
    assert task[0]["sequences"][0]["proteinChain"]["templates"] == []


def test_inference_rejects_persistent_cache_config(tmp_path):
    with pytest.raises(ValueError, match="prot_template_cache_dir"):
        InferenceDataset(dataset_config(tmp_path, str(tmp_path / "old_cache")))


def test_finalizer_rejects_injected_cached_featurizer(tmp_path):
    cached = TemplateHitFeaturizer(mmcif_dir=str(tmp_path),
        template_cache_dir=str(tmp_path / "old_cache"), fetch_remote=False)
    hits = tmp_path / "hits.a3m"
    hits.write_text(">2lwu_A/1-6 mol:protein length:6 fixture\nAAAACA\n")
    with pytest.raises(ValueError, match="prot_template_cache_dir"):
        finalize_template_hits("AAAAAA", hits, cached)


@pytest.mark.parametrize("cache", [None, ""])
def test_uncached_factory_reads_current_cif(tmp_path, monkeypatch, cache):
    configure(tmp_path, monkeypatch, cache)
    resolver = batch_inference._create_template_finalizer_featurizer(None, "2021-09-30")
    cif = tmp_path / "2lwu.cif"
    cif.write_text("old CIF text")
    assert resolver._hit_processor._fetch_or_read_cif("2lwu") == "old CIF text"
    cif.write_text("replacement CIF text")
    assert resolver._hit_processor._fetch_or_read_cif("2lwu") == "replacement CIF text"


@pytest.mark.parametrize("cache", [None, ""])
def test_uncached_inference_initialization_still_works(tmp_path, cache):
    dataset = InferenceDataset(dataset_config(tmp_path, cache))
    assert dataset.inputs[0]["name"] == "job"
