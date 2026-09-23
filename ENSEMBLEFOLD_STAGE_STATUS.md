# EnsembleFold wrapper stage status — 2026-09-23

Defaults: `write_input_json=true`, `compress_fold_input=false`, and
`compress_full_confidence=false`. Full-data emission remains independently
controlled by `need_atom_confidence` and is off by default. The obsolete
`write_now` option was removed; outputs are still written promptly.
See `docs/ensemblefold_prepared_inputs.md` for the public job-list schema.

CPU/offline regression: 257 tests and 82 subtests, with network-dependent tests
excluded/skipped; no GPU/model equivalence claim.

The proposed mandatory template declaration was cancelled; existing omission/
null and `use_template` behavior is retained. Legacy
`msa.precomputed_msa_dir` is still accepted but is not recommended for new tasks.
Do not mix it with explicit paired/unpaired fields: publication can change which
legacy channel is used. This compatibility risk was accepted as unused in the
current workflow, not repaired. Old `templatesPath` remains unsupported.
External `FILE_` ligand portability and multi-rank output contention remain
limitations. Current prepared input is not provenance for previously skipped
models produced under other conditions.
