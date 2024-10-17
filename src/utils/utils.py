import os
import multiprocessing as mp

AVAILABLE_DATASETS = {
    "domain_drift": "Domain Drift",
    "cifar10_4070": "CIFAR-10 (40-70)",
    "cifar10_5592": "CIFAR-10 (55-92)",
    "imagenet": "ImageNet",
    "imagenet_v2_matched-frequency": "ImageNet V2 M-F",
    "imagenet_v2_threshold-0.7": "ImageNet V2 T-0.7",
    "imagenet_pytorch_models": "ImageNet PyTorch",
    "imagenet_v2_top-images": "ImageNet V2 T-I",
    "emotion_detection": "Emotion Detection",
    "pacs": "PACS"
}


def get_available_cpus():
    slurm_env_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_env_cpus is not None:
        return int(slurm_env_cpus)
    return mp.cpu_count()