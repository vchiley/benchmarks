import mcli
from mcli.sdk import get_run_logs


CLUSTER_DESC = "r1z1"
GPU_TYPE = "A100_80GB"
GPU_AVAILABLE_FLOPS = 312_000_000_000_000
# RUN_NAMES = [
#     "thruput-v0-mosaic-gpt-760m-8xa100-80gb-s2048-b256-bf16-ikvP4T",
#     "thruput-v0-mosaic-gpt-350m-8xa100-80gb-s2048-b256-bf16-RP1cY5",
# ]
RUN_NAMES = [
    "thruput-v0mosaic-gpt-350m-sdata-VcpVfY",
    "thruput-v0mosaic-gpt-350m1-GMfNen",
    "thruput-v0mosaic-gpt-760m-sdata-rzCwcS",
    "thruput-v0mosaic-gpt-760m1-CnTED0",
    "thruput-v0mosaic-gpt-1b-sdata-ynUk2k",
]


def extract_from_loglines(string, lines):
    for line in lines:
        if f"{string}: " in line[:len(f"{string}: ")]:
            return line.split(' ')[-1]

    raise ValueError(f'{string=} not found in log')


print(f"| Model | ParamCount | Cluster Description | NumGPUs | GPUType | SeqLen | GlobalBatchSize (Tokens) | GlobalBatchSize (Samples) | MicroBatchSize | Precision | Throughput | MFU |")
for run_name in RUN_NAMES:
    try:
        log = get_run_logs(run_name)
        lines = log.split('\n')

        precision = extract_from_loglines("precision", lines)
        global_train_batch_size = int(extract_from_loglines("global_train_batch_size", lines))
        seq_len = int(extract_from_loglines("max_seq_len", lines))
        global_batchsize_tokens = global_train_batch_size * seq_len
        n_gpus = int(extract_from_loglines("n_gpus", lines))
        n_params = int(extract_from_loglines("n_params", lines))

        for line in lines[-25:]:
            if "throughput/samples_per_sec" in line:
                throughput = float(line.split(' ')[-1])

        for line in lines[-25:]:
            if "trainer/grad_accum" in line:
                grad_accum = int(line.split(' ')[-1])

        micro_batchsize = global_train_batch_size // n_gpus // grad_accum


        # mfu is approximated using thoughtput and param count
        # the number of paramters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # there are 3 passes of a NN (fwd, bwd, delta) - we multiply by 3 ie 2 * 3 * n_param
        # this gets us FLOPs / token
        flops_per_token = 2 * 3 * n_params
        flops_per_seq = flops_per_token * seq_len
        mfu = flops_per_seq * throughput / (n_gpus * GPU_AVAILABLE_FLOPS)

        print(f"| mosaic_gpt | {n_params} | {CLUSTER_DESC} | {n_gpus} | {GPU_TYPE} | {seq_len} | {global_batchsize_tokens} | {global_train_batch_size} | {micro_batchsize} | {precision} | {throughput} | {mfu} |")
    except mcli.api.exceptions.MAPIException:
        print(f"run {run_name} logs not loaded")
    except:
        print(f"run {run_name} logs loaded but not parsed")
