"""Failed work must reach the CLI without discarding successful predictions."""

import json
from types import SimpleNamespace

import pytest
import torch
from click.testing import CliRunner
from ml_collections import ConfigDict

from protenix.utils.input_json import load_input_json
from runner import batch_inference, inference


def _write_jobs(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([
        {"name": name, "modelSeeds": [7, 8], "sequences": []}
        for name in names
    ]))
    return path


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    """Replace only model/data/CIF work; exercise the real failure propagation."""
    output = tmp_path / "output"
    error_dir = output / "ERR"
    error_dir.mkdir(parents=True)
    configs = ConfigDict(dict(
        input_json_path="", seeds=[7, 8], use_seeds_in_json=False,
        deterministic=False, dump_dir=str(output), skip=False,
        model_name="protenix_base_default_v1.0.0",
        skip_amp=dict(confidence_head=True, sample_diffusion=True),
    ))
    state = SimpleNamespace(mode="predict", visited=[])

    def fails(name):
        return name.startswith("bad") or (
            name == "seed_job" and torch.initial_seed() == 7
        )

    class Loader:
        def __init__(self, jobs):
            self.dataset = jobs

        def __iter__(self):
            for index, job in enumerate(self.dataset):
                name = job["name"]
                state.visited.append((name, torch.initial_seed()))
                data = dict(
                    sample_name=name, sample_index=index,
                    entity_poly_type={"1": "polypeptide(L)"},
                    **{key: torch.tensor(1) for key in
                       ("N_asym", "N_token", "N_atom", "N_msa")},
                )
                error = "fixture data error" if state.mode == "data" and fails(name) else ""
                yield [(data, None, error)]

    def predict(data):
        if state.mode == "predict" and fails(data["sample_name"]):
            raise RuntimeError("fixture prediction error")
        return {"coordinate": torch.zeros((1, 1, 3))}

    def dump(*, pdb_id, seed, **kwargs):
        if state.mode == "dump" and fails(pdb_id):
            raise RuntimeError("fixture writer error")
        path = output / pdb_id / "models" / f"seed-{seed}_sample-0_model.cif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data_success\n")

    runner = SimpleNamespace(
        configs=configs, error_dir=str(error_dir), predict=predict,
        update_model_configs=lambda value: None,
        dumper=SimpleNamespace(dump=dump),
    )
    monkeypatch.setattr(inference, "get_inference_dataloader",
                        lambda configs: Loader(load_input_json(configs.input_json_path)))
    monkeypatch.setattr(inference, "fix_cterminal_carboxyl_oxygens",
                        lambda coordinates, atoms: coordinates)
    monkeypatch.setattr(batch_inference, "get_default_runner", lambda **kwargs: runner)
    monkeypatch.setattr(batch_inference, "init_logging", lambda: None)
    return runner, state, output


@pytest.mark.parametrize("mode", ["predict", "data", "dump"])
def test_partial_seed_failure_raises_after_later_seed_is_saved(tmp_path, fake_model, mode):
    runner, state, output = fake_model
    state.mode = mode
    runner.configs.input_json_path = str(_write_jobs(tmp_path / "job.json", ["seed_job"]))

    with pytest.raises(RuntimeError):
        inference.infer_predict(runner, runner.configs)

    assert state.visited == [("seed_job", 7), ("seed_job", 8)]
    assert (output / "seed_job/models/seed-8_sample-0_model.cif").read_text() == "data_success\n"
    assert not (output / "seed_job/models/seed-7_sample-0_model.cif").exists()
    assert "fixture" in (output / "ERR/seed_job.txt").read_text()


@pytest.mark.parametrize("layout", ["seed", "jobs", "files", "invalid_file"])
def test_cli_reports_partial_failure_and_keeps_later_success(tmp_path, fake_model, layout):
    _, _, output = fake_model
    if layout == "seed":
        source = _write_jobs(tmp_path / "input.json", ["seed_job"])
        good = "seed_job"
    elif layout == "jobs":
        source = _write_jobs(tmp_path / "input.json", ["bad_job", "good_job"])
        good = "good_job"
    else:
        source = tmp_path / "inputs"
        _write_jobs(source / "01_bad.json", ["bad_job"])
        _write_jobs(source / "02_good.json", ["good_job"])
        if layout == "invalid_file":
            (source / "01_bad.json").write_text("not json")
        good = "good_job"

    result = CliRunner().invoke(batch_inference.predict, [
        "--input", str(source), "--out_dir", str(output),
        "--run_data_pipeline", "false", "--run_inference", "true",
        "--write_input_json", "false",
    ])

    assert result.exit_code != 0, result.output
    assert isinstance(result.exception, RuntimeError), repr(result.exception)
    assert (output / good / "models/seed-8_sample-0_model.cif").read_text() == "data_success\n"
    assert not (output / ".protenix_tmp").exists()


def test_cli_all_success_returns_zero(tmp_path, fake_model):
    _, _, output = fake_model
    source = _write_jobs(tmp_path / "input.json", ["good_job"])
    result = CliRunner().invoke(batch_inference.predict, [
        "--input", str(source), "--out_dir", str(output),
        "--run_data_pipeline", "false", "--write_input_json", "false",
    ])
    assert result.exit_code == 0, repr(result.exception)
    assert (output / "good_job/models/seed-7_sample-0_model.cif").exists()
    assert (output / "good_job/models/seed-8_sample-0_model.cif").exists()


def test_all_failed_still_raises(tmp_path, fake_model):
    runner, _, _ = fake_model
    runner.configs.input_json_path = str(_write_jobs(tmp_path / "job.json", ["bad_job"]))
    with pytest.raises(RuntimeError, match="no successful predictions"):
        inference.infer_predict(runner, runner.configs)


@pytest.mark.parametrize(
    ("assigned_names", "expected_error", "expected_visits"),
    [
        (["complete"], None, [("complete", 7), ("complete", 8)]),
        (
            ["complete", "bad_job"],
            "no successful predictions",
            [
                ("complete", 7),
                ("bad_job", 7),
                ("complete", 8),
                ("bad_job", 8),
            ],
        ),
        ([], "no successful predictions", []),
    ],
)
def test_direct_resume_distinguishes_skip_only_assigned_slice_from_no_success(
    tmp_path,
    fake_model,
    monkeypatch,
    assigned_names,
    expected_error,
    expected_visits,
):
    runner, _, output = fake_model
    jobs_path = _write_jobs(tmp_path / "jobs.json", ["complete", "bad_job"])
    runner.configs.input_json_path = str(jobs_path)
    runner.configs.skip = True
    runner.configs.sample_diffusion = {"N_sample": 1}
    runner.configs.model = {"N_model_seed": 1}
    runner.configs.need_atom_confidence = False
    visits = []

    class AssignedSliceLoader:
        def __init__(self, jobs):
            self.dataset = jobs

        def __iter__(self):
            for index, job in enumerate(self.dataset):
                name = job["name"]
                if name not in assigned_names:
                    continue
                visits.append((name, torch.initial_seed()))
                data = dict(
                    sample_name=name,
                    sample_index=index,
                    entity_poly_type={"1": "polypeptide(L)"},
                    **{
                        key: torch.tensor(1)
                        for key in ("N_asym", "N_token", "N_atom", "N_msa")
                    },
                )
                yield [(data, None, "")]

    monkeypatch.setattr(
        inference,
        "get_inference_dataloader",
        lambda configs: AssignedSliceLoader(load_input_json(configs.input_json_path)),
    )
    monkeypatch.setattr(
        "protenix.utils.prediction_resume.incomplete_model_seeds",
        lambda _output, job_name, seeds, *_args, **_kwargs: (
            [] if job_name == "complete" else list(seeds)
        ),
    )

    if expected_error is None:
        inference.infer_predict(runner, runner.configs)
        assert not (output / "complete/models").exists()
    else:
        with pytest.raises(RuntimeError, match=expected_error):
            inference.infer_predict(runner, runner.configs)

    assert visits == expected_visits


@pytest.mark.parametrize("damage", ["missing", "empty"])
def test_direct_resume_preserves_complete_job_seeds_and_reruns_all_samples(
    tmp_path, fake_model, damage
):
    runner, _, output = fake_model
    runner.configs.input_json_path = str(_write_jobs(
        tmp_path / "jobs.json", ["complete", "partial"]
    ))
    runner.configs.skip = True
    runner.configs.sample_diffusion = {"N_sample": 2}
    runner.configs.model = {"N_model_seed": 2}
    runner.configs.need_atom_confidence = False
    preserved = {}
    for job in ("complete", "partial"):
        for seed in (7, 8):
            for sample in range(4):
                prefix = f"seed-{seed}_sample-{sample}"
                for folder, suffix in (("models", "model.cif"),
                                       ("summary_confidences", "summary_confidences.json")):
                    path = output / job / folder / f"{prefix}_{suffix}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('data_model' if folder == 'models' else '{"score":1}')
                    if job == "complete" or seed == 7:
                        preserved[path] = (path.read_bytes(), path.stat().st_mtime_ns)
    damaged = output / "partial/summary_confidences/seed-8_sample-3_summary_confidences.json"
    if damage == "missing":
        damaged.unlink()
    else:
        damaged.write_bytes(b"")
    predicted = []
    dumped = []

    def predict(data):
        predicted.append((data["sample_name"], torch.initial_seed(),
                          runner.configs.sample_diffusion.N_sample,
                          runner.configs.model.N_model_seed))
        return {"coordinate": torch.zeros((4, 1, 3))}

    def dump(*, pdb_id, seed, pred_dict, **_kwargs):
        dumped.append((pdb_id, seed, pred_dict["coordinate"].shape[0]))

    runner.predict = predict
    runner.dumper.dump = dump
    inference.infer_predict(runner, runner.configs)
    assert predicted == [("partial", 8, 2, 2)]
    assert dumped == [("partial", 8, 4)]
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == before
               for path, before in preserved.items())
