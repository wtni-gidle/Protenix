"""Native-selected hits -> single-chain inline entries -> model features."""
import io
import json
from unittest import mock

import numpy as np
import pytest
from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from protenix.data.template.template_finalizer import (
    FinalizedTemplateResult, finalize_template_hits, validate_template_mapping,
)
from protenix.data.template.template_parser import TemplateHit, TemplateParser, TemplateSearchResult
from protenix.data.template.template_utils import TemplateHitFeaturizer
from protenix.data.template.single_chain import parse_single_chain
from protenix.utils.input_json import load_input_json, load_inline_templates, write_prepared_input_jsons
from runner.template_search import _finalize_template_path, update_template_info
from tests.test_inline_prepared_contract import cif_missing


def multichain(cif):
    data = MMCIF2Dict(io.StringIO(cif))
    for key in list(data):
        if key.startswith("_atom_site."):
            values = data[key]
            second = (["B"] * len(values) if key in ("_atom_site.label_asym_id", "_atom_site.auth_asym_id")
                      else [str(int(v) + len(values)) for v in values] if key == "_atom_site.id"
                      else list(values))
            data[key] = values + second
        elif key.startswith("_struct_asym."):
            data[key] += ["B"] if key == "_struct_asym.id" else data[key][:1]
    writer = PDB.MMCIFIO()
    writer.set_dict(data)
    buffer = io.StringIO()
    writer.save(buffer)
    return buffer.getvalue()


def hit(query, template_sequence, name="2lwu_A"):
    return TemplateHit(index=1, name=name, aligned_cols=3, sum_probs=7.0,
                       query=query, hit_sequence=template_sequence,
                       indices_query=[0, 1, 2], indices_hit=[3, 4, 5])


def selected_featurizer(cif, selected, domains):
    # Mock only native search selection. CIF extraction and consumers are real.
    features = [{"template_domain_names": np.array(domain.encode(), dtype=object),
                 "template_sum_probs": [0.9 - i / 10],
                 "template_release_date": np.array(b"2020-01-02", dtype=object)}
                for i, domain in enumerate(domains)]
    featurizer = mock.Mock()
    featurizer._template_cache_dir = None
    featurizer._max_hits = 2
    featurizer.get_templates.return_value = (TemplateSearchResult(features, selected, ["skipped hit"], ["fallback chain"]), {"load": 0.25})
    featurizer._hit_processor._fetch_or_read_cif.return_value = cif
    return featurizer


@pytest.mark.parametrize("suffix,parser", [(".a3m", "HmmsearchA3MParser.parse"), (".hhr", "HHRParser.parse")])
def test_native_selection_order_max_hits_and_metadata(tmp_path, cif_missing, suffix, parser):
    obj, _ = parse_single_chain(cif_missing)
    seq = obj.chain_to_seqres["A"]
    selected = [hit("AAA", seq, "3ccc_A"), hit("AAA", seq, "1aaa_A")]
    featurizer = selected_featurizer(cif_missing, selected, ["3ccc_A", "1aaa_A"])
    source = tmp_path / ("hits" + suffix)
    source.write_text("external search output")
    with mock.patch("protenix.data.template.template_finalizer." + parser, return_value=selected) as parse:
        result = finalize_template_hits("AAA", source, featurizer, sequence_uid="job_A", max_template_date="2021-09-30")
    parse.assert_called_once()
    assert [t["domainName"] for t in result.templates] == ["3ccc_A", "1aaa_A"]
    assert [t["sumProbability"] for t in result.templates] == [0.9, 0.8]
    assert all(t["releaseDate"] == "2020-01-02" and "chainId" not in t for t in result.templates)
    assert result.errors == ["skipped hit"] and result.warnings == ["fallback chain"]
    assert featurizer._max_hits == 2
    assert featurizer.get_templates.call_args.kwargs["max_template_date"] == "2021-09-30"


@pytest.mark.parametrize("compressed", [False, True])
def test_native_multichain_features_equal_single_chain_prepared_roundtrip(tmp_path, cif_missing, compressed):
    original = multichain(cif_missing)
    obj = TemplateParser.parse(file_id="2lwu", mmcif_string=original).mmcif_object
    seq = obj.chain_to_seqres["B"]
    # Original hit requested A; native selection actually fell back to B.
    selected = hit("AAA", seq)
    featurizer = selected_featurizer(original, [selected], ["2lwu_B"])
    source = tmp_path / "hits.a3m"
    source.write_text("external search output")
    with mock.patch("protenix.data.template.template_finalizer._parse_hits", return_value=[selected]):
        result = finalize_template_hits("AAA", source, featurizer)
    exported, chain = parse_single_chain(result.templates[0]["mmcif"])
    assert chain == "B" and exported.chain_to_seqres[chain] == seq
    assert result.templates[0]["templateIndices"] == [3, 4, 5]
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps([{"name": "job", "sequences": [{"proteinChain": {
        "sequence": "AAA", "count": 2, "templates": result.templates}}]}]))
    path = write_prepared_input_jsons(input_json, tmp_path / "out", compress_fold_input=compressed)[0]
    source.unlink()
    protein = load_input_json(path)[0]["sequences"][0]["proteinChain"]
    consumer = TemplateHitFeaturizer(mmcif_dir="", fetch_remote=False, kalign_binary_path=None)
    with mock.patch.object(consumer._hit_processor, "_fetch_or_read_cif", side_effect=AssertionError("No database during inference")):
        feature = consumer.parse_json_templates(load_inline_templates(protein["templates"]), "AAA").features[0]
    native, _ = consumer._hit_processor._extract_template_features(obj, "2lwu", {0: 3, 1: 4, 2: 5}, seq, "AAA", "B", True)
    for key in ("template_all_atom_positions", "template_all_atom_masks", "template_aatype", "template_sequence", "template_domain_names"):
        np.testing.assert_array_equal(feature[key], native[key], err_msg=key)
    assert feature["template_all_atom_masks"][1].sum() == 0
    assert "templatesPath" not in protein


@pytest.mark.parametrize("q,t", [([0, 1], [0]), ([-1], [0]), ([0], [-1]), ([2], [0]), ([0], [2]), ([0, 0], [0, 1]), ([0, 1], [0, 0]), ([True], [0]), ([], [])])
def test_invalid_template_mappings_are_rejected(q, t):
    with pytest.raises(ValueError):
        validate_template_mapping(q, t, query_length=2, template_length=2)


@pytest.mark.parametrize("errors,fatal", [(["bad mapping"], True), (["Template 2lwu date unknown."], False), (["Template 2lwu date unknown.", "bad mapping"], True), ([], False)])
def test_no_hit_and_native_diagnostics_policy(errors, fatal):
    protein = {"sequence": "AAA"}
    result = FinalizedTemplateResult([], errors, ["candidate rejected"], {})
    with mock.patch("protenix.data.template.template_finalizer.finalize_template_hits", return_value=result):
        kwargs = dict(protein_chain=protein, templates_path="private.a3m", template_featurizer=object(), sequence_uid="job_A", max_template_date="2021-09-30")
        if fatal:
            with pytest.raises(RuntimeError, match="no usable templates"):
                _finalize_template_path(**kwargs)
            assert "templates" not in protein
        else:
            _finalize_template_path(**kwargs)
            assert protein["templates"] == []


@pytest.mark.parametrize("nonempty", [False, True])
def test_explicit_templates_never_search_or_create_database(tmp_path, cif_missing, nonempty):
    templates = [{"mmcif": cif_missing, "queryIndices": [0], "templateIndices": [0]}] if nonempty else []
    jobs = [{"name": "job", "sequences": [{"proteinChain": {"sequence": "AAA", "templates": templates}}]}]
    with mock.patch("runner.template_search.run_template_search", side_effect=AssertionError("No search")):
        assert not update_template_info(jobs, template_featurizer_factory=mock.Mock(side_effect=AssertionError("No database")))
    assert jobs[0]["sequences"][0]["proteinChain"]["templates"] == templates


def test_multichain_explicit_input_is_rejected(cif_missing):
    with pytest.raises(ValueError, match="exactly one"):
        parse_single_chain(multichain(cif_missing))
