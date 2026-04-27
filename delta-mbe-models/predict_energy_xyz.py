#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from ase.io import read
from torch_geometric.data import Data
from torch_geometric.transforms import Distance

import hydragnn
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    get_radius_graph,
    get_radius_graph_pbc,
)
from hydragnn.utils.distributed import get_device
from hydragnn.utils.input_config_parsing.config_utils import get_log_name_config
from hydragnn.utils.model import print_model


def load_minmax(dataset_path: Path):
    import pickle

    with dataset_path.open("rb") as handle:
        minmax_node = pickle.load(handle)
        minmax_graph = pickle.load(handle)
    return minmax_node, minmax_graph


def load_dataset_from_pickle(dataset_path: Path):
    import pickle

    with dataset_path.open("rb") as handle:
        _ = pickle.load(handle)
        _ = pickle.load(handle)
        dataset = pickle.load(handle)
    return dataset


def compute_max_edge_length(dataset, radius, max_neighbours, use_pbc):
    if use_pbc:
        for data in dataset:
            data.pbc = [True, True, True]
        compute_edges = get_radius_graph_pbc(
            radius=radius,
            loop=False,
            max_neighbours=max_neighbours,
        )
    else:
        compute_edges = get_radius_graph(
            radius=radius,
            loop=False,
            max_neighbours=max_neighbours,
        )

    compute_edge_lengths = Distance(norm=False, cat=True)

    max_edge_length = torch.tensor(float("-inf"))
    for data in dataset:
        data = compute_edges(data)
        if not use_pbc:
            data = compute_edge_lengths(data)
        max_edge_length = torch.max(max_edge_length, torch.max(data.edge_attr))

    if max_edge_length.item() <= 0:
        raise ValueError("Failed to compute a valid max edge length from training data.")

    return max_edge_length


def build_data_from_xyz(xyz_path: Path):
    atoms = read(xyz_path, parallel=False)
    pos = torch.tensor(atoms.arrays["positions"], dtype=torch.float32)
    z = torch.tensor(atoms.arrays["numbers"], dtype=torch.float32).unsqueeze(-1)
    return Data(pos=pos, x=z)


def prepare_graph(data, radius, max_neighbours, use_pbc, max_edge_length):
    if use_pbc:
        data.pbc = [True, True, True]
        compute_edges = get_radius_graph_pbc(
            radius=radius,
            loop=False,
            max_neighbours=max_neighbours,
        )
        data = compute_edges(data)
    else:
        compute_edges = get_radius_graph(
            radius=radius,
            loop=False,
            max_neighbours=max_neighbours,
        )
        data = compute_edges(data)
        data = Distance(norm=False, cat=True)(data)
        data.edge_attr = data.edge_attr / max_edge_length
    return data


def load_model(model, model_path: Path):
    checkpoint = torch.load(model_path, map_location=get_device())
    state_dict = checkpoint["model_state_dict"]
    if next(iter(state_dict)).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Predict energy for a single XYZ file using a trained HydraGNN model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("hydra_energy.json"),
        help="Path to HydraGNN config JSON",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/serialized_dataset"),
        help="Directory containing serialized datasets",
    )
    parser.add_argument(
        "--xyz",
        type=Path,
        required=True,
        help="Input XYZ file to run prediction on",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name (defaults to log name derived from config)",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    dataset_name = config["Dataset"]["name"]

    config["NeuralNetwork"]["Architecture"].setdefault(
        "periodic_boundary_conditions", False
    )
    config["NeuralNetwork"]["Architecture"].setdefault("pe_dim", 0)
    config["NeuralNetwork"]["Architecture"]["input_dim"] = len(
        config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
    )

    train_path = dataset_dir / f"{dataset_name}-trainset.pkl"
    minmax_node, minmax_graph = load_minmax(train_path)

    use_pbc = config["NeuralNetwork"]["Architecture"][
        "periodic_boundary_conditions"
    ]
    radius = config["NeuralNetwork"]["Architecture"]["radius"]
    max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

    trainset = load_dataset_from_pickle(train_path)
    max_edge_length = compute_max_edge_length(
        trainset, radius=radius, max_neighbours=max_neighbours, use_pbc=use_pbc
    )

    data = build_data_from_xyz(args.xyz.expanduser().resolve())
    data = prepare_graph(
        data,
        radius=radius,
        max_neighbours=max_neighbours,
        use_pbc=use_pbc,
        max_edge_length=max_edge_length,
    )

    device = get_device()
    data = data.to(device)

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )
    print_model(model)

    model_name = args.model_name or get_log_name_config(config)
    model_path = Path("logs") / model_name / f"{model_name}.pk"
    load_model(model, model_path)

    model.to(device)
    model.eval()

    with torch.no_grad():
        predicted = model(data)

    energy = predicted[0].flatten().cpu()
    if config["NeuralNetwork"]["Variables_of_interest"].get("denormalize_output", False):
        ymin, ymax = minmax_graph[:, 0]
        energy = energy * (ymax - ymin) + ymin

    print(f"Predicted energy: {energy.item():.6f} eV")


if __name__ == "__main__":
    main()
