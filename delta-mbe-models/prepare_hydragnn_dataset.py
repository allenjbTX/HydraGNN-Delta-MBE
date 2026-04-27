#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.datasets.serializeddataset import SerializedWriter


def load_npz(npz_path: Path):
    arrays = np.load(npz_path)
    required = {"R", "Q", "E", "Z", "N"}
    missing = required.difference(arrays.files)
    if missing:
        raise KeyError(f"Missing required arrays in {npz_path}: {sorted(missing)}")
    return arrays


def build_dataset(arrays):
    positions = arrays["R"]
    energies = arrays["E"]
    atomic_numbers = arrays["Z"]
    num_atoms = arrays["N"]

    dataset = []
    for idx in range(positions.shape[0]):
        natoms = int(num_atoms[idx])
        pos = torch.tensor(positions[idx][:natoms], dtype=torch.float32)
        z = torch.tensor(atomic_numbers[idx][:natoms], dtype=torch.float32).unsqueeze(-1)
        energy = torch.tensor([energies[idx]], dtype=torch.float32)

        data = Data(pos=pos, x=z, y=energy)
        dataset.append(data)

    return dataset


def compute_minmax(dataset):
    node_values = torch.cat([data.x for data in dataset], dim=0)
    node_min = node_values.min(dim=0).values.numpy()
    node_max = node_values.max(dim=0).values.numpy()
    minmax_node = np.stack([node_min, node_max])

    graph_values = torch.cat([data.y for data in dataset], dim=0)
    graph_min = graph_values.min().item()
    graph_max = graph_values.max().item()
    minmax_graph = np.array([[graph_min], [graph_max]], dtype=np.float32)

    return minmax_node, minmax_graph


def main():
    parser = argparse.ArgumentParser(
        description="Convert water cluster NPZ data to HydraGNN serialized datasets."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path(
            "order_2.npz"
        ),
        help="Path to npz dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("serialized_datasets"),
        help="Directory to write serialized pickle datasets",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="order_2",
        help="Dataset name used in serialized filenames",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.9,
        help="Fraction of data used for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for splitting",
    )
    args = parser.parse_args()

    npz_path = args.npz.expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    arrays = load_npz(npz_path)
    dataset = build_dataset(arrays)
    minmax_node, minmax_graph = compute_minmax(dataset)

    torch.manual_seed(args.seed)
    trainset, valset, testset = split_dataset(
        dataset=dataset,
        perc_train=args.train_frac,
        stratify_splitting=False,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    SerializedWriter(
        trainset,
        str(output_dir),
        args.dataset_name,
        "trainset",
        minmax_node_feature=minmax_node,
        minmax_graph_feature=minmax_graph,
    )
    SerializedWriter(
        valset,
        str(output_dir),
        args.dataset_name,
        "valset",
    )
    SerializedWriter(
        testset,
        str(output_dir),
        args.dataset_name,
        "testset",
    )

    print(
        "Wrote serialized datasets:",
        len(trainset),
        len(valset),
        len(testset),
    )


if __name__ == "__main__":
    main()
