#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Run the EnsembleFold Protenix wrapper.

Usage:
  run_protenix.sh -i INPUT -o OUTPUT [options] [-- extra protenix options]

Required:
  -i PATH   Input JSON file or directory.
  -o PATH   Output collection root.

Common options:
  -d IDS    CUDA device IDs, for example 0.                       [0]
  -D BOOL   Run the data pipeline.                                [true]
  -P BOOL   Run model inference.                                  [true]
  -r LIST   One seed or comma-separated model seeds.
  -c INT    Pairformer cycles.                                    [10]
  -p INT    Diffusion steps.                                      [200]
  -e INT    Samples per seed.                                     [5]
  -t NAME   Inference dtype.                                      [bf16]
  -n NAME   Model checkpoint name. [protenix_base_default_v1.0.0]
  -M BOOL   Use protein MSA features/search.                       [true]
  -T BOOL   Enable template processing.                            [false]
  -R BOOL   Enable RNA MSA processing.                             [false]
  -w VALUE  Write prepared JSON: auto, true, or false.             [auto]
  -z BOOL   Compress prepared MSA/template resources.              [true]
  -S BOOL   Skip seeds whose required outputs already exist.      [false]
  -h        Show this help.

With -w auto, prepared JSON is written for data-only/combined runs and is not
rewritten for inference-only runs. Template/RNA database options can be passed
after -- using their native Protenix names.

Environment:
  PROTENIX_ENV_ACTIVATE  Optional path to a venv/conda activate script.
  PROTENIX_BIN           Optional path to the protenix executable.

Examples:
  # Data only: writes <name>_data.json and *.a3m.zst.
  run_protenix.sh -i seq.json -o result -D true -P false

  # Inference only with several seeds.
  run_protenix.sh -i result/seq/seq_data.json -o result \
    -D false -P true -r 101,102,103 -S true

  # Template search/finalisation with native database arguments.
  run_protenix.sh -i seq.json -o result -D true -P false -T true -- \
    --kalign_binary_path /path/to/kalign \
    --hmmsearch_binary_path /path/to/hmmsearch \
    --hmmbuild_binary_path /path/to/hmmbuild \
    --seqres_database_path /path/to/pdb_seqres.fasta
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

normalize_bool() {
    case "$2" in
        [Tt][Rr][Uu][Ee]) printf 'true' ;;
        [Ff][Aa][Ll][Ss][Ee]) printf 'false' ;;
        *) die "$1 must be true or false, got: $2" ;;
    esac
}

input_path=""
output_dir=""
gpu_devices="0"
run_data_pipeline="true"
run_inference="true"
model_seeds=""
cycle="10"
step="200"
sample="5"
dtype="bf16"
model_name="protenix_base_default_v1.0.0"
use_msa="true"
use_template="false"
use_rna_msa="false"
write_input_json="auto"
compress_fold_input="true"
skip="false"

while getopts ":i:o:d:D:P:r:c:p:e:t:n:M:T:R:w:z:S:h" opt; do
    case "$opt" in
        i) input_path="$OPTARG" ;;
        o) output_dir="$OPTARG" ;;
        d) gpu_devices="$OPTARG" ;;
        D) run_data_pipeline="$OPTARG" ;;
        P) run_inference="$OPTARG" ;;
        r) model_seeds="$OPTARG" ;;
        c) cycle="$OPTARG" ;;
        p) step="$OPTARG" ;;
        e) sample="$OPTARG" ;;
        t) dtype="$OPTARG" ;;
        n) model_name="$OPTARG" ;;
        M) use_msa="$OPTARG" ;;
        T) use_template="$OPTARG" ;;
        R) use_rna_msa="$OPTARG" ;;
        w) write_input_json="$OPTARG" ;;
        z) compress_fold_input="$OPTARG" ;;
        S) skip="$OPTARG" ;;
        h) usage; exit 0 ;;
        :) die "Option -$OPTARG requires a value." ;;
        \?) die "Unknown option: -$OPTARG" ;;
    esac
done
shift $((OPTIND - 1))
if [[ "${1:-}" == "--" ]]; then
    shift
fi

[[ -n "$input_path" ]] || { usage >&2; die "-i is required."; }
[[ -n "$output_dir" ]] || { usage >&2; die "-o is required."; }
[[ -e "$input_path" ]] || die "Input does not exist: $input_path"
run_data_pipeline="$(normalize_bool "-D" "$run_data_pipeline")"
run_inference="$(normalize_bool "-P" "$run_inference")"
use_msa="$(normalize_bool "-M" "$use_msa")"
use_template="$(normalize_bool "-T" "$use_template")"
use_rna_msa="$(normalize_bool "-R" "$use_rna_msa")"
compress_fold_input="$(normalize_bool "-z" "$compress_fold_input")"
skip="$(normalize_bool "-S" "$skip")"
if [[ "$run_data_pipeline" == "false" && "$run_inference" == "false" ]]; then
    die "At least one of -D or -P must be true."
fi
case "$write_input_json" in
    auto|AUTO|Auto)
        if [[ "$run_data_pipeline" == "false" && "$run_inference" == "true" ]]; then
            write_input_json="false"
        else
            write_input_json="true"
        fi
        ;;
    [Tt][Rr][Uu][Ee]) write_input_json="true" ;;
    [Ff][Aa][Ll][Ss][Ee]) write_input_json="false" ;;
    *) die "-w must be auto, true, or false, got: $write_input_json" ;;
esac

if [[ -n "${PROTENIX_ENV_ACTIVATE:-}" ]]; then
    [[ -f "$PROTENIX_ENV_ACTIVATE" ]] || die \
        "Activation script not found: $PROTENIX_ENV_ACTIVATE"
    # shellcheck disable=SC1090
    source "$PROTENIX_ENV_ACTIVATE"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
protenix_bin="${PROTENIX_BIN:-}"
if [[ -z "$protenix_bin" ]]; then
    for candidate in "$script_dir/.venv/bin/protenix" "$script_dir/venv/bin/protenix"; do
        if [[ -x "$candidate" ]]; then
            protenix_bin="$candidate"
            break
        fi
    done
fi
if [[ -z "$protenix_bin" ]]; then
    protenix_bin="$(command -v protenix || true)"
fi
[[ -n "$protenix_bin" && -x "$protenix_bin" ]] || die \
    "protenix executable not found; activate its environment or set PROTENIX_BIN."

cmd=(
    "$protenix_bin" pred
    --input "$input_path"
    --out_dir "$output_dir"
    --run_data_pipeline "$run_data_pipeline"
    --run_inference "$run_inference"
    --write_input_json "$write_input_json"
    --compress_fold_input "$compress_fold_input"
    --cycle "$cycle"
    --step "$step"
    --sample "$sample"
    --dtype "$dtype"
    --model_name "$model_name"
    --use_msa "$use_msa"
    --use_template "$use_template"
    --use_rna_msa "$use_rna_msa"
    --skip "$skip"
)
if [[ -n "$model_seeds" ]]; then
    cmd+=(--model_seeds "$model_seeds")
fi
if [[ "$run_inference" == "true" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_devices"
fi
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi

echo "Protenix command:"
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
