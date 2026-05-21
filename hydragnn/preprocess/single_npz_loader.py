##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import logging
import os

import numpy as np
import torch
from torch_geometric.data import Data

from hydragnn.preprocess.raw_dataset_loader import AbstractRawDataLoader
from hydragnn.utils.distributed import comm_reduce
from hydragnn.utils.model.model import tensor_divide

# WARNING: DO NOT use collective communication calls here because only rank 0
# uses these routines (unless dist=True is explicitly passed).

# Atomic numbers for the elements present in the 7ccn delta-MLIP dataset.
_ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "Lu": 71}

# Layout of data.x (node feature matrix):
#   col 0: atomic number
#   col 1: electrostatic potential (ESP)
#
# data.energy: (1,) delta energy — used by compute_grad_energy (not data.y)
# data.forces: (num_atoms, 3) delta force components — used by compute_grad_energy


class NpzDataLoader(AbstractRawDataLoader):
    """Dataset loader for a directory of per-structure .npz files.

    Each .npz file must contain arrays for a single structure:
        qm_coords : (num_atoms, 3)
        qm_elems  : (num_atoms,)   element symbols, e.g. 'C', 'H', 'Lu'
        esp       : (num_atoms,)
        delta_e   : (1,)
        delta_f   : (num_atoms, 3)
        dft_e     : (1,)
        dft_f     : (num_atoms, 3)

    The ``path`` entries in the config must point to directories containing
    these .npz files (one file per structure), matching the expectation of
    the base-class ``load_raw_data``.

    Expected config layout
    ----------------------
    {
        "name": "delta_mlip",
        "format": "npz",
        "path": {
            "total": "/path/to/npz_directory"
        },
        "node_features": {
            "name": ["atomic_number", "esp"],
            "dim":  [1, 1],
            "column_index": [0, 1]
        },
        "graph_features": {
            "name": [],
            "dim":  [],
            "column_index": []
        }
    }
    """

    def __init__(self, config, dist=False):
        super().__init__(config, dist)

    def load_raw_data(self):
        super().load_raw_data()
        serialized_dir = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset"
        for serial_data_name in self.serial_data_name_list:
            stats_path = os.path.join(
                serialized_dir,
                serial_data_name.replace(".pkl", "_energy_forces_stats.npz"),
            )
            np.savez(
                stats_path,
                minmax_energy=self.minmax_energy,
                minmax_forces=self.minmax_forces,
                node_feature_mean=self.node_feature_mean,
                node_feature_std=self.node_feature_std,
            )

    def normalize_dataset(self):
        # ------------------------------------------------------------------
        # Node-feature normalization: z-score (standardization) per column.
        #
        # We deliberately do NOT call super().normalize_dataset(), which
        # min-max scales each node-feature column to [0, 1]. Min-max is a poor
        # fit here: the single Lu atom (Z=71) sets the atomic_number maximum, so
        # the light atoms (H, C, N, O -> Z 1..8) are squashed into [0, 0.10]
        # with almost no resolution, and the ESP gets squeezed too. Standardizing
        # each column to zero mean / unit variance keeps the light atoms well
        # separated (Lu becomes a clear positive outlier) and centers the ESP.
        #
        # NOTE: minmax_node_feature is metadata only downstream (it is pickled
        # but never re-applied to data.x), so storing the standardization stats
        # in it is safe and mirrors the minmax_energy = [[mean],[std]] convention
        # used below.
        # ------------------------------------------------------------------
        num_graph_features = len(self.graph_feature_dim)

        all_x = torch.cat(
            [data.x for dataset in self.dataset_list for data in dataset], dim=0
        ).double()
        n = float(all_x.shape[0])
        feat_sum = all_x.sum(dim=0)
        feat_sumsq = (all_x ** 2).sum(dim=0)

        if self.dist:
            n = comm_reduce(
                np.array([n], dtype=np.float64), torch.distributed.ReduceOp.SUM
            )[0]
            feat_sum = torch.tensor(
                comm_reduce(feat_sum.numpy(), torch.distributed.ReduceOp.SUM)
            )
            feat_sumsq = torch.tensor(
                comm_reduce(feat_sumsq.numpy(), torch.distributed.ReduceOp.SUM)
            )

        feat_mean = feat_sum / n
        feat_var = feat_sumsq / n - feat_mean ** 2
        feat_std = torch.sqrt(torch.clamp(feat_var, min=0.0))
        # Guard against constant columns (avoid divide-by-zero).
        feat_std = torch.where(feat_std > 1e-12, feat_std, torch.ones_like(feat_std))

        dtype = torch.get_default_dtype()
        self.node_feature_mean = feat_mean.cpu().numpy()
        self.node_feature_std = feat_std.cpu().numpy()

        logging.info(
            "NpzDataLoader node-feature standardization: mean=%s, std=%s",
            np.array2string(self.node_feature_mean, precision=6),
            np.array2string(self.node_feature_std, precision=6),
        )

        feat_mean_t = feat_mean.to(dtype)
        feat_std_t = feat_std.to(dtype)
        for dataset in self.dataset_list:
            for data in dataset:
                data.x = ((data.x.to(dtype) - feat_mean_t) / feat_std_t).to(dtype)

        # Populate the attributes the downstream pipeline expects.
        self.minmax_node_feature = np.vstack(
            [self.node_feature_mean, self.node_feature_std]
        )
        self.minmax_graph_feature = np.full((2, num_graph_features), np.nan)

        energies = np.array(
            [data.energy.item() for dataset in self.dataset_list for data in dataset]
        )
        e_mean = float(energies.mean())
        e_std = float(energies.std())

        if self.dist:
            # Welford-style distributed mean/std via sum and sum-of-squares
            n = len(energies)
            s = energies.sum()
            ss = (energies ** 2).sum()
            n = comm_reduce(np.array([n], dtype=np.float64), torch.distributed.ReduceOp.SUM)[0]
            s = comm_reduce(np.array([s]), torch.distributed.ReduceOp.SUM)[0]
            ss = comm_reduce(np.array([ss]), torch.distributed.ReduceOp.SUM)[0]
            e_mean = s / n
            e_std = float(np.sqrt(ss / n - e_mean ** 2))

        logging.info(
            "NpzDataLoader z-score normalization: e_mean=%.6f, e_std=%.6f (N=%d)",
            e_mean, e_std, len(energies),
        )

        # Store stats so load_raw_data can write them to disk.
        # Convention: minmax_energy[0] = mean, minmax_energy[1] = std
        self.minmax_energy = np.array([[e_mean], [e_std]])
        # Forces share the energy std so F_target = F_true/e_std is
        # consistent with F_pred = -dE_norm/dr.
        self.minmax_forces = np.array([[0.0], [e_std]])

        dtype = torch.get_default_dtype()
        for dataset in self.dataset_list:
            for data in dataset:
                data.energy = torch.tensor(
                    [(data.energy.item() - e_mean) / e_std], dtype=dtype
                )
                data.forces = (data.forces / e_std).to(dtype)

    def transform_input_to_data_object_base(self, filepath):
        if not filepath.endswith(".npz"):
            return None

        raw = np.load(filepath)

        coords = raw["qm_coords"]          # (num_atoms, 3)
        elems = raw["qm_elems"]            # (num_atoms,)
        delta_e = float(raw["delta_e"])
        delta_f = raw["delta_f"]           # (num_atoms, 3)
        #dft_e = float(raw["dft_e"])
        #dft_f = raw["dft_f"]               # (num_atoms, 3)
        esp = raw["esp"]                   # (num_atoms,)

        atomic_nums = np.array(
            [_ATOMIC_NUMBERS[e.strip()] for e in elems], dtype=np.float64
        ).reshape(-1, 1)

        # x: [atomic_number | esp]
        x = np.concatenate([atomic_nums, esp.reshape(-1, 1)], axis=1)

        data = Data()
        dtype = torch.get_default_dtype()
        data.pos = torch.tensor(coords, dtype=dtype)
        data.x = torch.tensor(x, dtype=dtype)
        data.energy = torch.tensor([delta_e], dtype=dtype)
        data.forces = torch.tensor(delta_f, dtype=dtype)
        return data
