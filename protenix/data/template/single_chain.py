"""Single-chain CIFs retain full polymer numbering, not only observed atoms."""
import io

import numpy as np
from Bio import PDB

from protenix.data.template.template_parser import TemplateParser


def parse_single_chain(mmcif: str, file_id: str = "template"):
    parsed = TemplateParser.parse(file_id=file_id, mmcif_string=mmcif)
    obj = parsed.mmcif_object
    if obj is None:
        raise ValueError(f"Template requires complete polymer sequence metadata: {parsed.errors}")
    proteins = TemplateParser._get_protein_chains(obj.raw_string)
    if len(proteins) != 1 or len(obj.chain_to_seqres) != 1:
        raise ValueError("Explicit template must contain exactly one protein chain")
    return obj, next(iter(obj.chain_to_seqres))


def extract_single_chain(obj, author_chain: str) -> str:
    """Filter native mmCIF tables without rewriting coordinates or seq positions.

    Same metadata-preserving strategy as EnsembleFold's OpenDDE exporter; no
    import from another method's repository or environment is required.
    """
    source = obj.raw_string
    proteins = TemplateParser._get_protein_chains(source)
    labels = {label for label, author in zip(source["_atom_site.label_asym_id"],
              source["_atom_site.auth_asym_id"], strict=True)
              if author == author_chain and label in proteins}
    if len(labels) != 1:
        raise ValueError(f"Cannot select one protein label chain for {author_chain!r}")
    label = next(iter(labels))
    atom_rows = [i for i, value in enumerate(source["_atom_site.label_asym_id"]) if value == label]
    entities = {source["_atom_site.label_entity_id"][i] for i in atom_rows}
    seq_rows = [i for i, value in enumerate(source["_entity_poly_seq.entity_id"]) if value in entities]
    monomers = {source["_entity_poly_seq.mon_id"][i] for i in seq_rows}
    output = {"data_": "single_template"}
    for key, values in source.items():
        if key.startswith(("_entry.", "_exptl.", "_pdbx_audit_revision_history.",
                           "_refine.", "_em_3d_reconstruction.", "_reflns.")):
            output[key] = list(values)

    def copy_rows(prefix, rows):
        for key, values in source.items():
            if key.startswith(prefix):
                output[key] = [values[i] for i in rows]

    copy_rows("_atom_site.", atom_rows)
    copy_rows("_struct_asym.", [i for i, value in enumerate(source["_struct_asym.id"]) if value == label])
    copy_rows("_entity_poly_seq.", seq_rows)
    for prefix, id_key, allowed in [
        ("_entity.", "_entity.id", entities),
        ("_entity_poly.", "_entity_poly.entity_id", entities),
        ("_chem_comp.", "_chem_comp.id", monomers),
        ("_pdbx_poly_seq_scheme.", "_pdbx_poly_seq_scheme.asym_id", {label}),
    ]:
        if id_key in source:
            copy_rows(prefix, [i for i, value in enumerate(source[id_key]) if value in allowed])
    if "_entity_poly.pdbx_strand_id" in output:
        output["_entity_poly.pdbx_strand_id"] = [author_chain] * len(output["_entity_poly.pdbx_strand_id"])
    writer = PDB.MMCIFIO()
    writer.set_dict(output)
    buffer = io.StringIO()
    writer.save(buffer)
    result = buffer.getvalue()
    rebuilt, rebuilt_chain = parse_single_chain(result)
    if obj.chain_to_seqres[author_chain] != rebuilt.chain_to_seqres[rebuilt_chain]:
        raise ValueError("Single-chain export changed the complete template sequence")
    # The same native atom extraction must see identical coordinates and masks.
    from protenix.data.template.template_utils import TemplateHitProcessor
    processor = TemplateHitProcessor(mmcif_dir="", kalign_binary_path=None)
    before = processor._get_atom_positions(obj, author_chain, 150.0, False)
    after = processor._get_atom_positions(rebuilt, rebuilt_chain, 150.0, False)
    for left, right in zip(before, after, strict=True):
        if not np.array_equal(left, right):
            raise ValueError("Single-chain export changed template coordinates or masks")
    return result
