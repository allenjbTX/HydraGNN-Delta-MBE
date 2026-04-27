#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from mpi4py import MPI
import torch
import torch.distributed as dist

import hydragnn
from hydragnn.preprocess.serialized_dataset_loader import SerializedDataLoader
from hydragnn.utils.input_config_parsing.config_utils import get_log_name_config
from hydragnn.utils.model import print_model
from hydragnn.utils.print.print_utils import log
from hydragnn.utils.profiling_and_tracing.time_utils import Timer


def load_minmax(dataset_path: Path):
    import pickle

    with dataset_path.open("rb") as handle:
        minmax_node = pickle.load(handle)
        minmax_graph = pickle.load(handle)
    return minmax_node, minmax_graph


def main():
    parser = argparse.ArgumentParser(description="Train HydraGNN on water cluster data.")
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
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    hydragnn.utils.print.setup_log(get_log_name_config(config))
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD

    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    dataset_dir = args.dataset_dir.expanduser().resolve()
    dataset_name = config["Dataset"]["name"]

    config["NeuralNetwork"]["Architecture"].setdefault(
        "periodic_boundary_conditions", False
    )
    config["NeuralNetwork"]["Architecture"].setdefault("pe_dim", 0)

    train_path = dataset_dir / f"{dataset_name}-trainset.pkl"
    val_path = dataset_dir / f"{dataset_name}-valset.pkl"
    test_path = dataset_dir / f"{dataset_name}-testset.pkl"

    minmax_node, minmax_graph = load_minmax(train_path)
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_node_feature"
    ] = minmax_node
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_graph_feature"
    ] = minmax_graph

    timer = Timer("load_data")
    timer.start()

    loader = SerializedDataLoader(config, dist=False)
    trainset = loader.load_serialized_data(str(train_path))
    valset = loader.load_serialized_data(str(val_path))
    testset = loader.load_serialized_data(str(test_path))

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    if rank == 0:
        print_model(model)

    comm.Barrier()

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer, verbosity
    )

    print_model(model)

    log_name = get_log_name_config(config)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    if dist.is_initialized():
        dist.barrier()

    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
