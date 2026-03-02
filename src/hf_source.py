from datasets import load_dataset
import random

def load_hf_rows(dataset_name: str, subset: str, split: str, max_rows: int, seed: int):
    ds = load_dataset(dataset_name, subset, split=split)
    n = len(ds)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[: min(max_rows, n)]
    return [ds[i] for i in idxs]