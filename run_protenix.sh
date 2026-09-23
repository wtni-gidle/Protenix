#!/bin/bash
# Exit immediately if a command fails.
set -e

# Allowed values: "current", "delta"
# current: use the protenix command from the environment that is already active.
# delta:   use the Protenix environment installed during the Delta tests.
server="current"

echo "Server: $server"

usage() {
    echo ""
    echo "Please make sure all required parameters are given"
    echo "Usage: $0 <OPTIONS>"
    echo "Required Parameters:"
    echo "-i <input_path>                 Input JSON file or a directory of inputs."
    echo "-o <output_dir>                 Directory in which results will be saved."
    echo "Optional Parameters:"
    echo "-d <gpu_device>                 CUDA device IDs, for example 0. (default: 0)"
    echo "-D <run_data_pipeline>          Run the data pipeline. (default: true)"
    echo "-P <run_inference>              Run model inference. (default: true)"
    echo "-r <model_seeds>                One seed or comma-separated seeds, e.g. 1,2,3."
    echo "-c <cycle>                      Number of Pairformer cycles. (default: 10)"
    echo "-p <step>                       Number of diffusion steps. (default: 200)"
    echo "-s <sample>                     Number of samples per seed. (default: 5)"
    echo "-t <dtype>                      Inference dtype. (default: bf16)"
    echo "-n <model_name>                 Checkpoint name. (default: protenix_base_default_v1.0.0)"
    echo "-M <use_msa>                    Build/use protein MSA features. (default: true)"
    echo "-T <use_template>               Build/use template features. (default: false)"
    echo "-m <max_template_date>          Latest template release date, YYYY-MM-DD. (default: 2021-09-30)"
    echo "-R <use_rna_msa>                Build/use RNA MSA features. (default: false)"
    echo "-w <write_input_json>           Write prepared JSON: true/false. (default: true)"
    echo "-z <compress_fold_input>        Write prepared resources as .zst. (default: false)"
    echo "-f <compress_full_confidence>   Write detailed confidence as compressed NPZ. (default: false)"
    echo "-S <skip>                       Skip seeds whose expected outputs exist. (default: false)"
    echo "-h                              Show this help."
    echo ""
    echo "Examples:"
    echo "  # Run only the data pipeline."
    echo "  $0 -i seq.json -o result -D true -P false"
    echo ""
    echo "  # Read seq_data.json and predict several seeds."
    echo "  $0 -i result/seq/seq_data.json -o result -D false -P true -r 101,102,103 -S true"
    exit 1
}

# region: Parse command line arguments
while getopts "i:o:d:D:P:r:c:p:s:t:n:M:T:m:R:w:z:f:S:h" opt; do
    case "${opt}" in
    i) input_path=$OPTARG ;;
    o) output_dir=$OPTARG ;;
    d) gpu_device=$OPTARG ;;
    D) run_data_pipeline=$OPTARG ;;
    P) run_inference=$OPTARG ;;
    r) model_seeds=$OPTARG ;;
    c) cycle=$OPTARG ;;
    p) step=$OPTARG ;;
    s) sample=$OPTARG ;;
    t) dtype=$OPTARG ;;
    n) model_name=$OPTARG ;;
    M) use_msa=$OPTARG ;;
    T) use_template=$OPTARG ;;
    m) max_template_date=$OPTARG ;;
    R) use_rna_msa=$OPTARG ;;
    w) write_input_json=$OPTARG ;;
    z) compress_fold_input=$OPTARG ;;
    f) compress_full_confidence=$OPTARG ;;
    S) skip=$OPTARG ;;
    h) usage ;;
    *) usage ;;
    esac
done
# endregion

# region: Check required parameters
if [[ "$input_path" == "" || "$output_dir" == "" ]]; then
    usage
fi

if [[ ! -e "$input_path" ]]; then
    echo "Error: input path does not exist: $input_path"
    exit 1
fi
# endregion

# region: Set default values
if [[ "$gpu_device" == "" ]]; then gpu_device="0"; fi
if [[ "$run_data_pipeline" == "" ]]; then run_data_pipeline="true"; fi
if [[ "$run_inference" == "" ]]; then run_inference="true"; fi
if [[ "$cycle" == "" ]]; then cycle="10"; fi
if [[ "$step" == "" ]]; then step="200"; fi
if [[ "$sample" == "" ]]; then sample="5"; fi
if [[ "$dtype" == "" ]]; then dtype="bf16"; fi
if [[ "$model_name" == "" ]]; then model_name="protenix_base_default_v1.0.0"; fi
if [[ "$use_msa" == "" ]]; then use_msa="true"; fi
if [[ "$use_template" == "" ]]; then use_template="false"; fi
if [[ "$max_template_date" == "" ]]; then max_template_date="2021-09-30"; fi
if [[ "$use_rna_msa" == "" ]]; then use_rna_msa="false"; fi
if [[ "$write_input_json" == "" ]]; then write_input_json="true"; fi
if [[ "$compress_fold_input" == "" ]]; then compress_fold_input="false"; fi
if [[ "$compress_full_confidence" == "" ]]; then compress_full_confidence="false"; fi
if [[ "$skip" == "" ]]; then skip="false"; fi

if [[ "$run_data_pipeline" == "false" && "$run_inference" == "false" ]]; then
    echo "Error: run_data_pipeline and run_inference cannot both be false."
    exit 1
fi

# endregion

# region: Set paths and activate the environment for each server
if [[ "$server" == "current" ]]; then
    # Activate your environment before running this script.
    protenix_bin="protenix"

elif [[ "$server" == "delta" ]]; then
    env_path=/work/hdd/bbgs/nwentao/protenix_wrapper_test/venv/bin/activate
    protenix_bin=/work/hdd/bbgs/nwentao/protenix_wrapper_test/venv/bin/protenix

    source "$env_path"
else
    echo "Error: server is invalid: $server"
    exit 1
fi
# endregion

if ! command -v "$protenix_bin" >/dev/null 2>&1; then
    echo "Error: Protenix executable does not exist: $protenix_bin"
    exit 1
fi

if [[ "$run_inference" == "true" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_device"
fi

#### Command arguments
command_args=(
    pred
    --input "$input_path"
    --out_dir "$output_dir"
    --run_data_pipeline "$run_data_pipeline"
    --run_inference "$run_inference"
    --write_input_json "$write_input_json"
    --compress_fold_input "$compress_fold_input"
    --compress_full_confidence "$compress_full_confidence"
    --cycle "$cycle"
    --step "$step"
    --sample "$sample"
    --dtype "$dtype"
    --model_name "$model_name"
    --use_msa "$use_msa"
    --use_template "$use_template"
    --max_template_date "$max_template_date"
    --use_rna_msa "$use_rna_msa"
    --skip "$skip"
)

if [[ "$model_seeds" != "" ]]; then
    command_args+=(--model_seeds "$model_seeds")
fi

# Run Protenix with the requested parameters.
echo "$protenix_bin ${command_args[*]}"
"$protenix_bin" "${command_args[@]}"
