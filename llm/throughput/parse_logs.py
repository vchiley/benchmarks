import argparse
import mcli
from mcli import sdk as msdk


GPU_AVAILABLE_FLOPS = 312_000_000_000_000


def parse_args():    
    parser = argparse.ArgumentParser(
        description="""
        Parse run configs to get MosaicGPT training throughput.
        MFU and HFU are defined in https://arxiv.org/abs/2205.05198
        All FLOP calculations do not include norm, act, residual, etc.
        """
    )

    parser.add_argument('--project', type=str, default='tput')
    parser.add_argument('--filters', type=str, default=[], nargs='+')

    return parser.parse_args()


def extract_from_loglines(string, lines):
    for line in lines:
        if f"{string}: " in line[:len(f"{string}: ")]:
            return line.split(' ')[-1]

    raise ValueError(f'{string=} not found in log')


def get_runs(args):
    runs = [r for r in msdk.get_runs() if args.project in r.name]
    for filter in args.filters:
        runs = [r for r in runs if filter in r.name]

    def sort_key(r):
        model_name = [s for s in r.name.split('-') if 'gpt' in s][0]
        if model_name[-1] == 'm':
            model_name_size = 1e6
        elif model_name[-1] == 'b':
            model_name_size = 1e9
        else:
            print(model_name)
            raise ValueError
        model_size = int(model_name[3:-1])
        return (model_name_size, model_size, r.config.parameters['max_seq_len'], r.config.parameters['global_train_batch_size'])
    runs.sort(reverse=True, key=sort_key)
    
    return runs


def filter_runs(runs):
    pop_runs = []
    for run in runs:
        if run.status == msdk.RunStatus('FAILED'):
            print(f"run {run.name} has FAILED (likely due to OOM error)")
            pop_runs.append(run)
        
    for run in pop_runs:
        runs.pop(runs.index(run))

    pop_runs = []
    for run in runs:
        if run.status in [
            msdk.RunStatus('FAILED_PULL'),
            msdk.RunStatus('PENDING'),
            msdk.RunStatus('QUEUED'),
            msdk.RunStatus('RUNNING'),
            msdk.RunStatus('SCHEDULED'),
            msdk.RunStatus('STARTING'),
            msdk.RunStatus('STOPPED'),
            msdk.RunStatus('STOPPING'),
            msdk.RunStatus('TERMINATING'),
        ]:
            print(f"run {run.name} has run status {run.status}")
            pop_runs.append(run)
    for run in pop_runs:
        runs.pop(runs.index(run))

    return runs


def parse_run(run):
    n_params = gpu_num = seq_len = global_batchsize_tokens = global_train_batch_size = micro_batchsize = precision = throughput = mfu = None

    model_name = [s for s in run.name.split('-') if 'gpt' in s][0]
    cluster = run.config.cluster
    gpu_num = run.config.gpu_num
    gpu_type = run.config.gpu_type

    precision = run.config.parameters['precision']
    seq_len = run.config.parameters['max_seq_len']
    global_train_batch_size = run.config.parameters['global_train_batch_size']


    # with MAPIConnection.get_current_connection():
    #     logs = msdk.get_run_logs(run)
    #     lines = []
    #     for line in logs:
    #         lines += line.split('\n')

    logs = msdk.get_run_logs(run)
    lines = []
    for line in logs:
        lines += line.split('\n')

    # MAPIConnection.get_current_connection().close()

    
    for line in lines[-25:]:
        if "trainer/grad_accum" in line:
            grad_accum = int(line.split(' ')[-1])
            break

    for line in lines:
        if f"n_params: " in line[:len(f"n_params: ")]:
            n_params = int(line.split(' ')[-1])
            break

    for line in lines[-25:]:
        if "throughput/samples_per_sec" in line:
            throughput = float(line.split(' ')[-1])
            break

    d_model = run.config.parameters['model']['d_model']
    n_heads = run.config.parameters['model']['n_heads']
    n_layers = run.config.parameters['model']['n_layers']

    global_batchsize_tokens = global_train_batch_size * seq_len
    micro_batchsize = global_train_batch_size // gpu_num // grad_accum

    # mfu is approximated using thoughtput and param count
    # the number of paramters is approximately the number of multiply-accumulates (MAC) in the network
    # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
    # there are 3 passes of a NN (fwd, bwd, delta) - we multiply by 3 ie 2 * 3 * n_param
    # this gets us FLOPs / token
    flops_per_token = 2 * n_params
    flops_per_seq = flops_per_token * seq_len
    mfu = 3 * flops_per_seq * throughput / (gpu_num * GPU_AVAILABLE_FLOPS)
    
    # there are 2 passes of a NN (fwd, bwd) - we multiply by 2 (no parameters to compute the gradient to)
    attn_flops_per_seq = n_layers * 2 * (d_model * (seq_len**2))
    mfu_w_attn = (3 * flops_per_seq + 2 * attn_flops_per_seq) * throughput / (gpu_num * GPU_AVAILABLE_FLOPS)

    mfu_w_recomp = 4 * flops_per_seq * throughput / (gpu_num * GPU_AVAILABLE_FLOPS)
    mfu_w_attn_w_recomp = (4 * flops_per_seq + 3 * attn_flops_per_seq) * throughput / (gpu_num * GPU_AVAILABLE_FLOPS)

    print(f"| {model_name: >7} | {n_params: >11} | {cluster: >7} | {gpu_num: >7} | {gpu_type: >7} | {seq_len: >6} | {global_batchsize_tokens: >19} | {global_train_batch_size: >19} | {micro_batchsize: >14} | {grad_accum: >9} | {precision: >9} | {throughput: >10.4f} | {mfu:.4f} | {mfu_w_attn:.4f} | {mfu_w_recomp:.4f} | {mfu_w_attn_w_recomp:.4f} |")


def main(args):
    runs = get_runs(args)
    runs = filter_runs(runs)

    print(
        "| Model   | ParamCount  | Cluster | NumGPUs | GPUType   | SeqLen | GlobalBatchSize (T) | GlobalBatchSize (S) | MicroBatchSize | GradAccum | Precision | Throughput | MFU**  | MFU    | HFU**  | HFU    |\n"
        "| ------- | ----------- | ------- | ------- | --------- | ------ | ------------------- | ------------------- | -------------- | --------- | --------- | ---------- | ------ | ------ | ------ | ------ |"
    )
    for run in runs:
        parse_run(run)


if __name__ == "__main__":
    args = parse_args()

    from mcli.api.engine.engine import MAPIConnection
    with MAPIConnection.get_current_connection():  # TODO: mcli.sdk has a bug; remove when bug is fixed
        main(args)
