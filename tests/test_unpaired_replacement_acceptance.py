"""Prepared unpaired replacement reaches real inference condition features.

Only checkpoint/model execution and external search are intercepted. The JSON
loader, atom/CCD feature builder, MSA pairing/deduplication, CIF parser, template
assembly, and InferenceDataset.process_one all run unchanged. This is acceptance
of the current consumers, not numerical equivalence with an upstream release.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ml_collections import ConfigDict

from protenix.data.inference.infer_dataloader import InferenceDataset
from protenix.utils.input_json import load_input_json, write_prepared_input_jsons
from protenix.utils.text_io import read_text, write_zstd_text_atomic
from tests.test_inline_prepared_contract import cif_missing


OLD_UNPAIRED = ">q\nAAAAAA\n>old\nACAAAA\n"
NEW_UNPAIRED = ">q\nAAAAAA\n>replacement\nAAggDAAA\n>duplicate\nAADAAA\n"
PAIRED_A = ">q\nAAAAAA\n>UniRef100_SHARED_TAX\nAAAAAC\n"
PAIRED_B = ">q\nCCCCCC\n>UniRef100_SHARED_TAX\nCCCCCA\n"


def config(path, output):
    return ConfigDict(dict(
        input_json_path=str(path), dump_dir=str(output), use_msa=True,
        msa_pair_as_unpair=True, use_rna_msa=False, use_template=True,
        esm=dict(enable=False), sample_diffusion=dict(guidance=dict(enable=False)),
    ))


def consume(dataset):
    data, _, _ = dataset.process_one(dataset.inputs[0])
    return {key: value.detach().cpu().numpy().copy()
            for key, value in data["input_feature_dict"].items()
            if key in {"msa", "profile", "deletion_mean", "has_deletion",
                       "prot_paired_num_alignments"}
            or key.startswith("template_")}


def prepare(tmp_path, cif, compress):
    # Two different six-residue proteins exercise native inter-chain pairing.
    # Reuse the real single-chain CIF fixture, including missing residue five.
    source = tmp_path / "source.json"
    template = dict(mmcif=cif, queryIndices=[0, 1, 2, 3, 4, 5],
                    templateIndices=[0, 1, 2, 3, 4, 5])
    proteins = [dict(sequence="AAAAAA", count=1, pairedMsa=PAIRED_A,
                     unpairedMsa=OLD_UNPAIRED, templates=[template]),
                dict(sequence="CCCCCC", count=1, pairedMsa=PAIRED_B,
                     unpairedMsa=">q\nCCCCCC\n", templates=[])]
    source.write_text(json.dumps([dict(name="replacement", modelSeeds=[17],
        sequences=[dict(proteinChain=protein) for protein in proteins])]))
    path = Path(write_prepared_input_jsons(source, tmp_path / "prepared",
                                         compress_fold_input=compress)[0])
    return path


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def assert_new_conditions(before, after):
    # Both hits share species TAX: native pairing must join AAAAAC + CCCCCA.
    # Check paired depth too: merged unpaired hits can coincide in one row.
    joined_pair = [0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 0]
    for features in (before, after):
        assert np.any(np.all(features["msa"] == joined_pair, axis=1))
        assert features["prot_paired_num_alignments"].item() >= 2
    old_row, new_row = [0, 4, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0]
    assert np.any(np.all(before["msa"][:, :6] == old_row, axis=1))
    assert not np.any(np.all(after["msa"][:, :6] == old_row, axis=1))
    assert np.any(np.all(after["msa"][:, :6] == new_row, axis=1))
    # Query + paired-as-unpaired + deduplicated replacement: hand-counted 1/3 D.
    assert before["profile"][2, 3] == 0
    assert after["profile"][2, 3] == pytest.approx(1 / 3)
    assert after["deletion_mean"][2] == pytest.approx(2 / 3)
    assert np.any(after["has_deletion"][:, 2])
    assert before["template_atom_mask"][0, 4].sum() == 0
    assert before["template_atom_mask"][0, 5].sum() > 0
    assert before["template_atom_positions"][0, 5].any()
    for key in before:
        if key.startswith("template_"):
            np.testing.assert_array_equal(after[key], before[key], err_msg=key)
    # Paired source bytes are checked separately. Do not freeze pool-dependent
    # native pairing/dedup tensors when the unpaired population changes.


@pytest.mark.parametrize("compress", [False, True])
def test_same_dataset_rereads_same_path_without_stale_condition_features(
    tmp_path, cif_missing, compress,
):
    path = prepare(tmp_path, cif_missing, compress)
    inline = consume(InferenceDataset(config(tmp_path / "source.json",
                                             tmp_path / "unused-output")))
    dataset = InferenceDataset(config(path, tmp_path / "unused-output"))
    before = consume(dataset)
    for key in inline:
        np.testing.assert_array_equal(before[key], inline[key], err_msg=key)
    protein = dataset.inputs[0]["sequences"][0]["proteinChain"]
    old_files = snapshot(path.parent)
    unpaired = Path(protein["unpairedMsaPath"])
    if compress:
        write_zstd_text_atomic(unpaired, NEW_UNPAIRED)
    else:
        unpaired.write_text(NEW_UNPAIRED)
    assert_new_conditions(before, consume(dataset))
    new_files = snapshot(path.parent)
    assert new_files.keys() == old_files.keys()
    changed = {name for name in old_files if old_files[name] != new_files[name]}
    assert changed == {str(unpaired.relative_to(path.parent))}
    assert read_text(protein["pairedMsaPath"]) == PAIRED_A


@pytest.mark.parametrize("data", [False, True])
@pytest.mark.parametrize("publish", [False, True])
def test_wrapper_replacement_reaches_model_conditions_and_obeys_publication(
    tmp_path, monkeypatch, cif_missing, data, publish,
):
    from runner import batch_inference, msa_search, template_search

    path = prepare(tmp_path, cif_missing, False)
    before = consume(InferenceDataset(config(path, tmp_path / "unused-output")))
    protein = load_input_json(path)[0]["sequences"][0]["proteinChain"]
    Path(protein["unpairedMsaPath"]).write_text(NEW_UNPAIRED)
    source_snapshot = snapshot(path.parent)
    output = tmp_path / "published"
    # An existing stale bundle detects false publication and true rewrite.
    write_prepared_input_jsons(tmp_path / "source.json", output,
                              compress_fold_input=False)
    published_before = snapshot(output)
    observed = []
    preprocess_calls = []
    real_preprocess = batch_inference.preprocess_input

    def preprocess(*args, **kwargs):
        preprocess_calls.append(True)
        return real_preprocess(*args, **kwargs)

    def forbidden_search(*args, **kwargs):
        pytest.fail("Complete explicit prepared conditions must not search")

    def infer_at_model_boundary(runner, runtime_config):
        # Same dataset consumer used by inference, with real atom/MSA/template
        # features. No model checkpoint is loaded and no prediction is claimed.
        dataset = InferenceDataset(ConfigDict(runtime_config))
        observed.append(consume(dataset))

    monkeypatch.setattr(batch_inference, "preprocess_input", preprocess)
    monkeypatch.setattr(msa_search, "msa_search", forbidden_search)
    monkeypatch.setattr(template_search, "run_hmmsearch_with_a3m", forbidden_search)
    monkeypatch.setattr(batch_inference, "get_default_runner", lambda **kwargs:
                        SimpleNamespace(configs=config(path, output)))
    monkeypatch.setattr(batch_inference, "_run_infer_predict", infer_at_model_boundary)
    batch_inference.inference_jsons(
        json_file=str(path), out_dir=str(output), run_data_pipeline=data,
        run_inference=True, write_input_json=publish, compress_fold_input=False,
        use_template=True, use_msa=True, use_rna_msa=False, skip=False,
    )
    assert len(observed) == 1
    assert bool(preprocess_calls) is data
    assert_new_conditions(before, observed[0])
    assert snapshot(path.parent) == source_snapshot
    if not publish:
        published_after = snapshot(output)  # Fresh recursive public inventory.
        assert published_after.keys() == published_before.keys()
        assert published_after == published_before
        return
    public = output / "replacement/replacement_data.json"
    raw = json.loads(public.read_text())[0]["sequences"][0]["proteinChain"]
    assert raw["unpairedMsaPath"] == "msas/replacement__A_unpairedmsa.a3m"
    assert raw["pairedMsaPath"] == "msas/replacement__A_pairedmsa.a3m"
    assert raw["templates"][0] == dict(
        mmcifPath="msas/replacement__A_template_0.cif",
        queryIndices=[0, 1, 2, 3, 4, 5], templateIndices=[0, 1, 2, 3, 4, 5])
    loaded = load_input_json(public)[0]["sequences"][0]["proteinChain"]
    assert read_text(loaded["unpairedMsaPath"]) == NEW_UNPAIRED
    assert read_text(loaded["pairedMsaPath"]) == PAIRED_A
    assert (Path(loaded["templates"][0]["mmcifPath"]).read_bytes()
            == Path(protein["templates"][0]["mmcifPath"]).read_bytes())
    # The rewritten relative bundle remains usable after relocation.
    moved = tmp_path / "moved"
    public.parent.rename(moved)
    assert_new_conditions(before, consume(InferenceDataset(
        config(moved / public.name, tmp_path / "unused-output"))))
