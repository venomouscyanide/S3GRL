import json

from torch_geometric.profile import get_stats_summary, get_model_size, count_parameters, get_data_size, \
    get_cpu_memory_from_gc, get_gpu_memory_from_gc
from torch_geometric.profile.utils import byte_to_megabyte


def profile_helper(all_stats, model, train_dataset, stats_suffix):
    summarized_stats = get_stats_summary(all_stats)
    model_size = get_model_size(model)
    parameters = count_parameters(model)
    train_dataset_size = get_data_size(train_dataset.data)
    cpu_usage = get_cpu_memory_from_gc()
    gpu_usage = get_gpu_memory_from_gc()
    stats = {}
    print("------------------------------------------")

    print(f"Summarized stats: {summarized_stats}")
    stats[
        'Average Time(in seconds)'
    ] = f'{summarized_stats.time_mean:.2f} ± {summarized_stats.time_std:.2f}'

    # Details about there params are here: https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html
    # we use all: combined statistics across all memory pools and peak: maximum value of this metric
    # for all below metrics and convert to megabytes

    # Returns the max of current GPU memory occupied by tensors in bytes for a given device.
    stats['Max Allocated CUDA (in MegaBytes)'] = f'{summarized_stats.max_allocated_cuda:.2f}'
    # Returns the max of current GPU memory managed by the caching allocator in bytes for a given device
    stats['Max Reserved CUDA (in MegaBytes)'] = f'{summarized_stats.max_reserved_cuda:.2f}'
    # amount of active memory in bytes.
    stats['Max Active CUDA (in MegaBytes)'] = f'{summarized_stats.max_active_cuda:.2f}'

    # from command: 'nvidia-smi --query-gpu=memory.free --format=csv'
    stats['Min NVIDIA SMI Free CUDA Memory (in MegaBytes)'] = f'{summarized_stats.min_nvidia_smi_free_cuda}'
    # from command: 'nvidia-smi --query-gpu=memory.used --format=csv'
    stats['Max NVIDIA SMI Used CUDA Memory (in MegaBytes)'] = f'{summarized_stats.max_nvidia_smi_used_cuda}'

    print("------------------------------------------")

    print(f"Model size: {model_size}")
    stats['Model size (in MegaBytes)'] = f'{byte_to_megabyte(model_size):.2f}'
    print(f"Parameters: {parameters}")
    stats['Number of Model Parameters'] = f'{parameters}'
    print(f"Train Dataset Size: {train_dataset_size}")
    stats['Train Dataset Size (in MegaByes)'] = f'{byte_to_megabyte(model_size):.2f}'

    print("------------------------------------------")

    print(f"CPU usage: {cpu_usage}")
    print(f"GPU usage: {gpu_usage}")

    print("------------------------------------------")
    with open(f'stats_{stats_suffix}.json', 'w') as stats_file:
        json.dump(stats, stats_file)
    print("fin profiling.")
