export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

FILE=$1
CONFIG_FILE=$2
TAG=$3
shift 3  # remove $1~$3 for $@

# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/ggm_setup/.cache/huggingface
export TORCH_HOME=/tmp/ggm_setup/.cache/torch
export NCCL_DEBUG=VERSION

PORTS=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
PORT=${PORTS[0]}

accelerate launch \
    --num_machines $ARNOLD_WORKER_NUM \
    --num_processes $(( $ARNOLD_WORKER_NUM * $ARNOLD_WORKER_GPU )) \
    --machine_rank $ARNOLD_ID \
    --main_process_ip $METIS_WORKER_0_HOST \
    --main_process_port $PORT \
    ${FILE} \
        --config_file ${CONFIG_FILE} \
        --tag ${TAG} \
        --pin_memory \
        --allow_tf32 \
$@
