"""Inline JSON template shape and hit contract (real CIF, no database)."""
import numpy as np
from protenix.data.constants import ATOM37_NUM
from protenix.data.template.template_utils import TemplateHitFeaturizer
from tests.test_inline_prepared_contract import cif_missing, entry


def test_model_feature_consumer_reloads_current_cif_and_copies_entities(tmp_path, cif_missing):
    from biotite.structure import AtomArray
    from protenix.data.template.template_featurizer import InferenceTemplateFeaturizer
    from protenix.data.template.single_chain import parse_single_chain
    seq = parse_single_chain(cif_missing)[0].chain_to_seqres["A"]
    n = len(seq)
    atoms = AtomArray(n * 2)
    atoms.set_annotation("asym_id_int", np.repeat([0, 1], n))
    atoms.set_annotation("chain_id", np.repeat(["A", "B"], n))
    atoms.set_annotation("res_id", np.tile(np.arange(1, n + 1), 2))
    atoms.set_annotation("centre_atom_mask", np.ones(n * 2, dtype=bool))
    resource = tmp_path / "template.cif"
    resource.write_text(cif_missing)
    assembly = [{"proteinChain": {"sequence": seq, "count": 2, "templates": [{
        "mmcifPath": str(resource), "queryIndices": list(range(n)),
        "templateIndices": list(range(n))}]}}]
    reader = TemplateHitFeaturizer(mmcif_dir="", fetch_remote=False)
    def consume():
        return InferenceTemplateFeaturizer.make_template_feature(
            assembly, atoms, use_template=True, online_template_featurizer=reader)
    before = consume()
    assert before["template_atom_mask"][0, 4].sum() == 0
    assert before["template_atom_mask"][0, 4 + n].sum() == 0
    np.testing.assert_array_equal(before["template_atom_mask"][0, :n], before["template_atom_mask"][0, n:])
    # Same path now contains a complete file. No stale feature/resource cache.
    from pathlib import Path
    resource.write_text((Path(__file__).parents[1] / "examples/2lwu.cif").read_text())
    after = consume()
    assert after["template_atom_mask"][0, 4].sum() > 0
    disabled = InferenceTemplateFeaturizer.make_template_feature(assembly, atoms, use_template=False, online_template_featurizer=reader)
    assert disabled["template_atom_mask"].sum() == 0


def test_json_template_parser(cif_missing):
    result = TemplateHitFeaturizer(mmcif_dir="", fetch_remote=False).parse_json_templates([entry(cif_missing)], "AAA")
    assert not result.errors
    assert len(result.features) == len(result.hits) == 1
    feature = result.features[0]
    assert feature["template_all_atom_positions"].shape == (3, ATOM37_NUM, 3)
    assert feature["template_all_atom_masks"].shape == (3, ATOM37_NUM)
    assert feature["template_aatype"].shape == (3,)
    assert np.sum(feature["template_all_atom_masks"]) > 0
    result_hit = result.hits[0]
    assert result_hit.query == "AAA" and result_hit.aligned_cols == 3
    assert result_hit.indices_query == [0, 1, 2] and result_hit.indices_hit == [3, 4, 5]
