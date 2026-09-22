"""Public prepared input behavior; network/model execution is not exercised."""
import io
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from protenix.data.template.template_parser import TemplateParser
from protenix.data.template.template_utils import TemplateHitFeaturizer
from protenix.utils.input_json import load_input_json, write_prepared_input_jsons
from protenix.utils.text_io import read_text
from runner import msa_search, template_search


@pytest.fixture
def cif_missing():
    source = MMCIF2Dict(str(Path(__file__).parents[1] / "examples/2lwu.cif"))
    # Full sequence remains unchanged; the fifth residue has no coordinates.
    keep = [i for i, seq in enumerate(source["_atom_site.label_seq_id"])
            if seq != "5" and source["_atom_site.pdbx_PDB_model_num"][i] == "1"]
    for key in source:
        if key.startswith("_atom_site."):
            source[key] = [source[key][i] for i in keep]
    writer = PDB.MMCIFIO()
    writer.set_dict(source)
    buffer = io.StringIO()
    writer.save(buffer)
    return buffer.getvalue()


def entry(cif):
    return {"mmcif": cif, "queryIndices": [0, 1, 2], "templateIndices": [3, 4, 5]}


def consumer(tmp_path):
    return TemplateHitFeaturizer(mmcif_dir=str(tmp_path), template_cache_dir=None,
                                kalign_binary_path=None, fetch_remote=False)


def test_explicit_single_chain_uses_full_sequence_missing_positions(tmp_path, cif_missing):
    parsed = consumer(tmp_path).parse_json_templates([entry(cif_missing)], "AAA")
    assert len(parsed.features) == 1
    assert parsed.features[0]["template_all_atom_masks"][1].sum() == 0
    assert parsed.features[0]["template_all_atom_masks"][[0, 2]].sum() > 0


@pytest.mark.parametrize("change", [{"queryIndices": []}, {"templateIndices": [999, 1, 2]},
                                     {"chainId": "A"}, {"queryIndices": [True, 1, 2]}])
def test_invalid_explicit_templates_raise_not_silently_drop(tmp_path, cif_missing, change):
    template = {**entry(cif_missing), **change}
    with pytest.raises(ValueError):
        consumer(tmp_path).parse_json_templates([template], "AAA")


def test_inline_bundle_has_no_sidecar_and_survives_move(tmp_path, cif_missing):
    source = tmp_path / "input.json"
    cif = tmp_path / "source.cif"
    cif.write_text(cif_missing)
    template = {**entry(cif_missing), "mmcifPath": "source.cif"}
    template.pop("mmcif")
    source.write_text(json.dumps([{"name": "job", "sequences": [{"proteinChain": {
        "sequence": "AAA", "count": 1, "templates": [template]}}]}]))
    output = Path(write_prepared_input_jsons(source, tmp_path / "out", compress_fold_input=True)[0])
    protein = json.loads(output.read_text())[0]["sequences"][0]["proteinChain"]
    assert protein["templates"][0]["mmcifPath"] == "msas/job__A_template_0.cif.zst"
    assert "chainId" not in protein["templates"][0]
    assert list((output.parent / "msas").glob("*.json")) == []
    cif.unlink()
    moved = tmp_path / "moved"
    output.parent.rename(moved)
    loaded = load_input_json(moved / "job_data.json")[0]["sequences"][0]["proteinChain"]
    resource = Path(loaded["templates"][0]["mmcifPath"])
    feature = consumer(tmp_path).parse_json_templates(
        [{**loaded["templates"][0], "mmcif": read_text(resource)}], "AAA").features[0]
    assert feature["template_all_atom_masks"][1].sum() == 0


def test_old_templates_path_is_rejected_before_resource_loading(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps([{"name": "job", "sequences": [{"proteinChain": {
        "sequence": "AAA", "count": 1, "templatesPath": "nonexistent.json"}}]}]))
    with pytest.raises(ValueError, match="templatesPath"):
        load_input_json(path)


def test_missing_msa_path_fails_instead_of_search(tmp_path):
    with pytest.raises(FileNotFoundError):
        msa_search.need_msa_search({"sequences": [{"proteinChain": {
            "sequence": "AAA", "pairedMsaPath": str(tmp_path / "missing")}}]})


def test_explicit_inline_empty_msa_does_not_search():
    assert not msa_search.need_msa_search({"sequences": [{"proteinChain": {
        "sequence": "AAA", "pairedMsa": "", "unpairedMsa": ""}}]})


def test_empty_msa_wins_over_legacy_directory_when_writing(tmp_path):
    (tmp_path / "pairing.a3m").write_text(">q\nAAA\n")
    (tmp_path / "non_pairing.a3m").write_text(">q\nAAA\n")
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{"name": "job", "sequences": [{"proteinChain": {
        "sequence": "AAA", "pairedMsa": "", "unpairedMsa": "",
        "msa": {"precomputed_msa_dir": str(tmp_path)}}}]}]))
    path = write_prepared_input_jsons(source, tmp_path / "out")[0]
    protein = load_input_json(path)[0]["sequences"][0]["proteinChain"]
    assert protein["pairedMsa"] == protein["unpairedMsa"] == ""
    assert "pairedMsaPath" not in protein and "unpairedMsaPath" not in protein


def test_joint_search_keeps_supplied_msa_and_only_fills_missing(tmp_path, monkeypatch):
    supplied = tmp_path / "deepmsa.a3m"
    supplied.write_text(">query\nAAA\n")
    job = {"sequences": [
        {"proteinChain": {"sequence": "AAA", "unpairedMsaPath": str(supplied), "pairedMsa": ""}},
        {"proteinChain": {"sequence": "CCC"}},
    ]}
    def server(sequences, output, mode):
        assert sequences == ["AAA", "CCC"]  # Preserve joint search context.
        paths = []
        for i, seq in enumerate(sequences):
            folder = tmp_path / str(i)
            folder.mkdir()
            for name in ["pairing.a3m", "non_pairing.a3m"]:
                (folder / name).write_text(f">q\n{seq}\n")
            paths.append(str(folder))
        return paths
    monkeypatch.setattr(msa_search, "msa_search", server)
    msa_search.update_seq_msa(job, str(tmp_path / "search"), "protenix")
    assert job["sequences"][0]["proteinChain"]["unpairedMsaPath"] == str(supplied)
    assert "pairedMsaPath" not in job["sequences"][0]["proteinChain"]
    assert Path(job["sequences"][1]["proteinChain"]["pairedMsaPath"]).is_file()


def test_template_search_reads_actual_paths_and_respects_explicit_empty(tmp_path, monkeypatch):
    paired = tmp_path / "custom_paired.a3m"
    unpaired = tmp_path / "custom_unpaired.a3m"
    paired.write_text(">q\nAAA\n")
    unpaired.write_text(">q\nAAA\n>new\nACA\n")
    jobs = [{"name": "job", "sequences": [{"proteinChain": {
        "sequence": "AAA", "pairedMsaPath": str(paired), "unpairedMsaPath": str(unpaired)}}]}]
    def hmm(**kwargs):
        assert kwargs["a3m"] == ">q\nAAA\n>q\nAAA\n>new\nACA\n"
        return ""
    monkeypatch.setattr(template_search, "run_hmmsearch_with_a3m", hmm)
    # No hits: normal no-template outcome, not a failed explicit input.
    from protenix.data.template import template_finalizer
    monkeypatch.setattr(template_finalizer, "finalize_template_hits", lambda **kwargs:
                        template_finalizer.FinalizedTemplateResult([], [], [], {}))
    kwargs = dict(out_dir=str(tmp_path / "scratch"), hmmsearch_binary_path=__file__,
                  hmmbuild_binary_path=__file__, seqres_database_path=__file__,
                  template_featurizer_factory=lambda: object())
    assert template_search.update_template_info(jobs, **kwargs)
    assert jobs[0]["sequences"][0]["proteinChain"]["templates"] == []
    assert "templatesPath" not in jobs[0]["sequences"][0]["proteinChain"]
    assert not template_search.update_template_info(jobs, **kwargs)
