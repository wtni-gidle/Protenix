import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from protenix.utils.input_json import load_input_json, write_prepared_input_jsons
from protenix.utils.text_io import read_text


def jobs(msa):
    return [{"name": "job", "modelSeeds": [1], "sequences": [{"proteinChain": {
        "sequence": "AAA", "count": 1, "pairedMsa": "", "unpairedMsaPath": str(msa),
        "templates": []}}]}]


def test_failed_bundle_publication_restores_resources(tmp_path, monkeypatch):
    from protenix.utils import input_json
    msa = tmp_path / "input.a3m"
    msa.write_text(">q\nAAA\n>new\nACA\n")
    source = tmp_path / "in.json"
    source.write_text(json.dumps(jobs(msa)))
    dest = tmp_path / "out/job"
    (dest / "msas").mkdir(parents=True)
    old = dest / "msas/job__A_unpairedmsa.a3m"
    old.write_text("previous MSA")
    old_json = dest / "job_data.json"
    old_json.write_text("previous JSON")
    replace = input_json.os.replace
    def fail(source, target):
        if Path(target) == old_json:
            raise OSError("injected publication failure")
        return replace(source, target)
    monkeypatch.setattr(input_json.os, "replace", fail)
    with pytest.raises(OSError):
        write_prepared_input_jsons(source, tmp_path / "out")
    assert old.read_text() == "previous MSA"
    assert old_json.read_text() == "previous JSON"


def test_inplace_republication_reads_swapped_sources_before_replacing(tmp_path):
    msa = tmp_path / "original.a3m"
    msa.write_text(">q\nAAA\n>u\nACA\n")
    source = tmp_path / "input.json"
    data = jobs(msa)
    data[0]["sequences"][0]["proteinChain"]["pairedMsa"] = ">q\nAAA\n>p\nAAC\n"
    source.write_text(json.dumps(data))
    path = Path(write_prepared_input_jsons(source, tmp_path / "out")[0])
    data = json.loads(path.read_text())
    protein = data[0]["sequences"][0]["proteinChain"]
    p, u = protein["pairedMsaPath"], protein["unpairedMsaPath"]
    protein["pairedMsaPath"], protein["unpairedMsaPath"] = u, p
    path.write_text(json.dumps(data))
    write_prepared_input_jsons(path, tmp_path / "out")
    result = load_input_json(path)[0]["sequences"][0]["proteinChain"]
    assert read_text(result["pairedMsaPath"]) == ">q\nAAA\n>u\nACA\n"
    assert read_text(result["unpairedMsaPath"]) == ">q\nAAA\n>p\nAAC\n"


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_combined_runtime_private_and_cleaned(tmp_path, monkeypatch, write, failure):
    from runner import batch_inference
    msa = tmp_path / "input.a3m"
    msa.write_text(">q\nAAA\n")
    source = tmp_path / "input.json"
    source.write_text(json.dumps(jobs(msa)))
    initial = source.read_bytes()
    output = tmp_path / "out"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    observed = []
    def preprocess(input_json, *, intermediate_json_path, out_dir, **kwargs):
        # Substitute expensive search only; real workflow chooses lifetime/location.
        destination = Path(intermediate_json_path)
        assert scratch in destination.parents
        assert output not in Path(out_dir).parents and Path(out_dir) != output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(load_input_json(input_json)))
        observed.append(destination)
        return str(destination)
    runner = SimpleNamespace(configs={})
    monkeypatch.setattr(batch_inference, "preprocess_input", preprocess)
    monkeypatch.setattr(batch_inference, "get_default_runner", lambda **kwargs: runner)
    def inference(runner, config):
        loaded = load_input_json(config["input_json_path"])
        assert read_text(loaded[0]["sequences"][0]["proteinChain"]["unpairedMsaPath"]) == ">q\nAAA\n"
        if failure:
            raise RuntimeError("injected inference failure")
    monkeypatch.setattr(batch_inference, "_run_infer_predict", inference)
    kwargs = dict(json_file=str(source), out_dir=str(output), run_data_pipeline=True,
                  run_inference=True, write_input_json=write, use_template=False)
    if failure:
        with pytest.raises(RuntimeError):
            batch_inference.inference_jsons(**kwargs)
    else:
        batch_inference.inference_jsons(**kwargs)
    assert observed
    assert all(not path.exists() for path in observed)
    assert list(scratch.iterdir()) == []
    assert source.read_bytes() == initial
    assert (output / "job/job_data.json").exists() == write
    assert not (output / ".protenix_tmp").exists()


def test_optional_esm_embeddings_use_private_runtime_not_cwd(tmp_path, monkeypatch):
    from ml_collections import ConfigDict
    from protenix.data.inference.infer_dataloader import InferenceDataset, ESMFeaturizer
    source = tmp_path / "in.json"
    source.write_text(json.dumps([{"name": "job", "sequences": []}]))
    scratch = tmp_path / "runtime"
    scratch.mkdir()
    config = ConfigDict(dict(input_json_path=str(source), dump_dir=str(tmp_path / "out"),
        use_msa=False, use_template=False, _runtime_dir=str(scratch), load_checkpoint_dir="unused",
        esm=dict(enable=True, model_name="test_esm", embedding_dim=8)))
    def compute(inputs, model_name, embedding_dir, sequence_fpath, checkpoint_dir):
        assert scratch in Path(embedding_dir).parents
        assert scratch in Path(sequence_fpath).parents
    monkeypatch.setattr(ESMFeaturizer, "precompute_esm_embedding", compute)
    monkeypatch.setattr(ESMFeaturizer, "__init__", lambda self, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    InferenceDataset(config)
    assert not (tmp_path / "esm_embeddings").exists()


def test_hmmbuild_and_kalign_scratch_respect_slurm(tmp_path, monkeypatch):
    from protenix.data.tools import search, kalign
    scratch = tmp_path / "slurm"
    scratch.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    def shell(command, *args, **kwargs):
        assert scratch in Path(command[-1]).parents
        Path(command[-2]).write_text("HMM")
    monkeypatch.setattr(search, "run_shell", shell)
    builder = search.Hmmbuild(binary_path="/bin/true")
    assert builder.build(">q\nAAAAAA\n", "afa") == "HMM"
    # Inspect the real alignment I/O boundary without executing a binary.
    class Process:
        def communicate(self):
            return b"", b""
        def wait(self):
            return 0
    def popen(command, **kwargs):
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        assert scratch in input_path.parents and scratch in output_path.parents
        output_path.write_text(">0\nAAAAAA\n>1\nAAAACA\n")
        return Process()
    monkeypatch.setattr(kalign.subprocess, "Popen", popen)
    aligner = object.__new__(kalign.Kalign)
    aligner.binary_path = "/bin/true"
    assert len(aligner.align(["AAAAAA", "AAAACA"])) == 2
    assert list(scratch.iterdir()) == []
