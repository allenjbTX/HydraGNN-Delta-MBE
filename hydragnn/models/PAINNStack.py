##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

# Adapted From the Following:
# Github: https://github.com/nityasagarjena/PaiNN-model/blob/main/PaiNN/model.py
# Paper: https://arxiv.org/pdf/2102.03150


from typing import Optional

import torch
from torch import nn
from torch_geometric import nn as geom_nn
from torch_geometric.nn import radius_graph
from torch.utils.checkpoint import checkpoint
from torch_geometric.typing import OptTensor

from .Base import Base
from hydragnn.utils.model.operations import get_edge_vectors_and_lengths

# Coulomb's constant in eV·Å/e² (matches positions in Å, charges in e, energy in eV).
_COULOMB_CONSTANT_EV_ANGSTROM = 14.399645351950543


class PAINNStack(Base):
    """
    Generates angles, distances, to/from indices, radial basis
    functions and spherical basis functions for learning.
    """

    def __init__(
        self,
        input_args,
        conv_args,
        edge_dim: int,
        num_radial: int,
        radius: float,
        *args,
        charge_conservation: bool = False,
        charge_head_index: Optional[int] = None,
        long_range_electrostatics: bool = False,
        long_range_cutoff: Optional[float] = None,
        coulomb_constant: float = _COULOMB_CONSTANT_EV_ANGSTROM,
        energy_head_index: int = 0,
        **kwargs
    ):
        self.edge_dim = edge_dim
        self.num_radial = num_radial
        self.radius = radius
        self.is_edge_model = True  # specify that mpnn cannot handle edge features
        self.charge_conservation = charge_conservation
        self.charge_head_index = charge_head_index
        if self.charge_conservation and self.charge_head_index is None:
            raise ValueError(
                "charge_conservation=True requires charge_head_index to be set "
                "to the index of the output head producing per-atom charges."
            )
        self.long_range_electrostatics = long_range_electrostatics
        self.long_range_cutoff = long_range_cutoff
        self.coulomb_constant = coulomb_constant
        self.energy_head_index = energy_head_index
        if self.long_range_electrostatics and self.charge_head_index is None:
            raise ValueError(
                "long_range_electrostatics=True requires charge_head_index to "
                "identify the output head producing per-atom charges."
            )
        super().__init__(input_args, conv_args, *args, **kwargs)

    @staticmethod
    def _apply_charge_conservation(charges: torch.Tensor, data) -> torch.Tensor:
        """Rescale per-atom partial charges so each molecule's charges sum to Q.

        Implements PhysNet eq 14: q_i_tilde = q_i - (sum_j q_j - Q) / N_atoms.
        ``data.total_charge`` is used as Q when present (shape ``(num_mol,)``
        or ``(num_mol, 1)``); otherwise the system is assumed neutral.
        """
        if charges.dim() == 1:
            charges = charges.unsqueeze(-1)
            squeeze_back = True
        else:
            squeeze_back = False

        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(charges.shape[0], dtype=torch.long, device=charges.device)
        )
        num_mol = int(batch.max().item()) + 1
        n_feat = charges.shape[1]

        q_sum = charges.new_zeros((num_mol, n_feat))
        q_sum.index_add_(0, batch, charges)

        n_atoms = torch.bincount(batch, minlength=num_mol).to(dtype=charges.dtype)

        q_ref = getattr(data, "total_charge", None)
        if q_ref is None:
            q_ref_mol = charges.new_zeros((num_mol, n_feat))
        else:
            q_ref_mol = q_ref.to(dtype=charges.dtype, device=charges.device).reshape(
                num_mol, n_feat
            )

        delta = (q_sum - q_ref_mol) / n_atoms.unsqueeze(-1)
        corrected = charges - delta[batch]
        return corrected.squeeze(-1) if squeeze_back else corrected

    @staticmethod
    def _phys_net_phi(r: torch.Tensor, r_cut: float) -> torch.Tensor:
        """PhysNet eq 8: smooth quintic cutoff (1 at r=0, 0 at r>=r_cut)."""
        ratio = r / r_cut
        inside = 1.0 - 6.0 * ratio ** 5 + 15.0 * ratio ** 4 - 10.0 * ratio ** 3
        return torch.where(r < r_cut, inside, torch.zeros_like(r))

    def _phys_net_chi(self, r: torch.Tensor) -> torch.Tensor:
        """PhysNet eq 13: damped Coulomb kernel.

        chi(r) = phi(2r) / sqrt(r^2 + 1) + (1 - phi(2r)) / r

        For r >= radius/2 this is exactly 1/r; for r < radius/2 it smoothly
        transitions to the damped form, removing the singularity at r=0.
        """
        phi = self._phys_net_phi(2.0 * r, self.radius)
        damped = 1.0 / torch.sqrt(r * r + 1.0)
        coulomb = 1.0 / r
        return phi * damped + (1.0 - phi) * coulomb

    def _compute_coulomb_energy(
        self, charges: torch.Tensor, data
    ) -> torch.Tensor:
        """Long-range Coulomb energy per molecule (PhysNet eq 12, no D3).

        Returns a tensor of shape ``(num_mol, 1)`` to match the energy head.
        """
        if charges.dim() == 1:
            charges = charges.unsqueeze(-1)

        pos = data.pos
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        )

        # Build an intra-molecule pair list. ``radius_graph`` already excludes
        # cross-graph pairs via ``batch`` and self-loops by default.
        cutoff = self.long_range_cutoff
        if cutoff is None or cutoff <= 0.0:
            # Effectively all-pairs: pick a cutoff larger than any plausible
            # molecular diameter so radius_graph returns every intra-graph pair.
            cutoff = float("inf")
            # radius_graph does not accept inf; fall back to a large finite value.
            with torch.no_grad():
                num_mol = int(batch.max().item()) + 1
                max_diam = 0.0
                for m in range(num_mol):
                    mask = batch == m
                    if mask.sum() < 2:
                        continue
                    pm = pos[mask]
                    d = torch.cdist(pm, pm).max().item()
                    if d > max_diam:
                        max_diam = d
                cutoff = max_diam + 1.0

        # max_num_neighbors must be large enough to retain every neighbor for
        # large molecules at long cutoff. Use the largest atom count we see.
        with torch.no_grad():
            max_n = int(torch.bincount(batch).max().item())

        edge_index = radius_graph(
            pos,
            r=float(cutoff),
            batch=batch,
            loop=False,
            max_num_neighbors=max(max_n - 1, 32),
        )
        # Keep each unordered pair only once.
        src, dst = edge_index[0], edge_index[1]
        mask = src < dst
        i, j = src[mask], dst[mask]

        num_mol = int(batch.max().item()) + 1
        if i.numel() == 0:
            return charges.new_zeros((num_mol, 1))

        r_ij = (pos[i] - pos[j]).norm(dim=-1)
        chi = self._phys_net_chi(r_ij)
        q_i = charges[i, 0]
        q_j = charges[j, 0]
        pair_e = self.coulomb_constant * q_i * q_j * chi  # (num_pairs,)

        mol_e = pair_e.new_zeros(num_mol)
        mol_e.index_add_(0, batch[i], pair_e)
        return mol_e.unsqueeze(-1)

    def forward(self, data):
        outputs = super().forward(data)
        if not (self.charge_conservation or self.long_range_electrostatics):
            return outputs

        # Base.forward returns either ``outputs`` or ``(outputs, outputs_var)``
        # depending on var_output; handle both.
        if isinstance(outputs, tuple):
            head_outputs, head_vars = outputs
        else:
            head_outputs, head_vars = outputs, None

        if self.charge_conservation:
            head_outputs[self.charge_head_index] = self._apply_charge_conservation(
                head_outputs[self.charge_head_index], data
            )

        if self.long_range_electrostatics:
            charges = head_outputs[self.charge_head_index]
            coulomb_e = self._compute_coulomb_energy(charges, data)
            energy = head_outputs[self.energy_head_index]
            # Match shape: energy is (num_mol, head_dim); coulomb_e is
            # (num_mol, 1). Broadcasting handles head_dim==1, the typical case.
            head_outputs[self.energy_head_index] = energy + coulomb_e

        if head_vars is not None:
            return head_outputs, head_vars
        return head_outputs

    def _init_conv(self):
        last_layer = 1 == self.num_conv_layers
        self.graph_convs.append(
            self._apply_global_attn(
                self.get_conv(
                    self.embed_dim,
                    self.hidden_dim,
                    last_layer,
                    edge_dim=self.edge_embed_dim,
                )
            )
        )
        self.feature_layers.append(nn.Identity())
        for i in range(self.num_conv_layers - 1):
            last_layer = i == self.num_conv_layers - 2
            self.graph_convs.append(
                self._apply_global_attn(
                    self.get_conv(
                        self.hidden_dim,
                        self.hidden_dim,
                        last_layer,
                        edge_dim=self.edge_embed_dim,
                    )
                )
            )
            self.feature_layers.append(nn.Identity())

    def get_conv(self, input_dim, output_dim, last_layer=False, edge_dim=None):
        hidden_dim = output_dim if input_dim == 1 else input_dim
        assert (
            hidden_dim > 1
        ), "PainnNet requires more than one hidden dimension between input_dim and output_dim."
        self_inter = PainnMessage(
            node_size=input_dim,
            num_radial=self.num_radial,
            cutoff=self.radius,
            edge_dim=edge_dim,
        )
        cross_inter = PainnUpdate(node_size=input_dim, last_layer=last_layer)
        """
        The following linear layers are to get the correct sizing of embeddings. This is
        necessary to use the hidden_dim, output_dim of HYDRAGNN's stacked conv layers correctly
        because node_scalar and node-vector are updated through a sum.
        """
        node_embed_out = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim),
        )  # Tanh activation is necessary to prevent exploding gradients when learning from random signals in test_graphs.py
        vec_embed_out = nn.Linear(input_dim, output_dim) if not last_layer else None

        if not last_layer:
            return geom_nn.Sequential(
                self.input_args,
                [
                    (
                        self_inter,
                        self.conv_args + " -> inv_node_feat, equiv_node_feat",
                    ),
                    (
                        cross_inter,
                        "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                    ),
                    (node_embed_out, "inv_node_feat -> inv_node_feat"),
                    (vec_embed_out, "equiv_node_feat -> equiv_node_feat"),
                    (
                        lambda inv_node_feat, equiv_node_feat: [
                            inv_node_feat,
                            equiv_node_feat,
                        ],
                        "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                    ),
                ],
            )
        else:
            return geom_nn.Sequential(
                self.input_args,
                [
                    (
                        self_inter,
                        self.conv_args + " -> inv_node_feat, equiv_node_feat",
                    ),
                    (
                        cross_inter,
                        "inv_node_feat, equiv_node_feat -> inv_node_feat",
                    ),  # v is not updated in the last layer to avoid hanging gradients
                    (
                        node_embed_out,
                        "inv_node_feat -> inv_node_feat",
                    ),  # No need to embed down v because it's not used anymore
                    (
                        lambda inv_node_feat, equiv_node_feat: [
                            inv_node_feat,
                            equiv_node_feat,
                        ],
                        "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                    ),
                ],
            )

    def _embedding(self, data):
        super()._embedding(data)

        assert (
            data.pos is not None
        ), "PAINN requires node positions (data.pos) to be set."

        # Get normalized edge vectors and lengths
        norm_edge_vec, edge_dist = get_edge_vectors_and_lengths(
            data.pos, data.edge_index, data.edge_shifts, normalize=True
        )

        conv_args = {
            "edge_index": data.edge_index.t().to(torch.long),
            "diff": norm_edge_vec,
            "dist": edge_dist,
        }

        if self.use_edge_attr:
            assert (
                data.edge_attr is not None
            ), "Data must have edge attributes if use_edge_attributes is set."
            conv_args.update({"edge_attr": data.edge_attr})

        if self.use_global_attn:
            # encode node positional embeddings
            x = self.pos_emb(data.pe)
            # if node features are available, genrate mebeddings, concatenate with positional embeddings and map to hidden dim
            if self.input_dim:
                x = torch.cat((self.node_emb(data.x.float()), x), 1)
                x = self.node_lin(x)
            # repeat for edge features and relative edge encodings
            if self.is_edge_model:
                e = self.rel_pos_emb(data.rel_pe)
                if self.use_edge_attr:
                    e = torch.cat((self.edge_emb(conv_args["edge_attr"]), e), 1)
                    e = self.edge_lin(e)
                conv_args.update({"edge_attr": e})
        else:
            x = data.x
        # Instantiate tensor to hold equivariant traits
        v = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        return x, v, conv_args


class PainnMessage(nn.Module):
    """Message function"""

    def __init__(self, node_size: int, num_radial: int, cutoff: float, edge_dim: int):
        super().__init__()

        self.node_size = node_size
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.edge_dim = edge_dim

        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )

        self.filter_layer = nn.Linear(num_radial, node_size * 3)

        if self.edge_dim is not None:
            self.edge_filter = nn.Sequential(
                nn.Linear(self.edge_dim, node_size),
                nn.SiLU(),
                nn.Linear(node_size, node_size * 3),
            )

    def forward(
        self,
        node_scalar,
        node_vector,
        edge,
        edge_diff,
        edge_dist,
        edge_attr: OptTensor = None,
    ):
        dtype = node_scalar.dtype
        if node_vector.dtype != dtype:
            node_vector = node_vector.to(dtype=dtype)
        if edge_dist.dtype != dtype:
            edge_dist = edge_dist.to(dtype=dtype)
        if edge_diff.dtype != dtype:
            edge_diff = edge_diff.to(dtype=dtype)
        if edge_attr is not None and edge_attr.dtype != dtype:
            edge_attr = edge_attr.to(dtype=dtype)
        # remember to use v_j, s_j but not v_i, s_i
        filter_weight = self.filter_layer(
            sinc_expansion(edge_dist, self.num_radial, self.cutoff)
        )
        filter_weight = filter_weight * cosine_cutoff(edge_dist, self.cutoff)
        if edge_attr is not None:
            filter_weight = filter_weight * self.edge_filter(edge_attr)

        scalar_out = self.scalar_message_mlp(node_scalar)
        filter_out = filter_weight * scalar_out[edge[:, 1]]

        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out,
            self.node_size,
            dim=1,
        )

        # num_pairs * 3 * node_size, num_pairs * node_size
        message_vector = node_vector[edge[:, 1]] * gate_state_vector.unsqueeze(1)
        edge_vector = gate_edge_vector.unsqueeze(1) * (edge_diff / edge_dist).unsqueeze(
            -1
        )
        message_vector = message_vector + edge_vector

        # sum message
        residual_scalar = torch.zeros_like(node_scalar)
        residual_vector = torch.zeros_like(node_vector)
        residual_scalar.index_add_(0, edge[:, 0], message_scalar)
        residual_vector.index_add_(0, edge[:, 0], message_vector)

        # new node state
        new_node_scalar = node_scalar + residual_scalar
        new_node_vector = node_vector + residual_vector

        return new_node_scalar, new_node_vector


class PainnUpdate(nn.Module):
    """Update function"""

    def __init__(self, node_size: int, last_layer=False):
        super().__init__()

        self.update_U = nn.Linear(node_size, node_size)
        self.update_V = nn.Linear(node_size, node_size)
        self.last_layer = last_layer

        if not self.last_layer:
            self.update_mlp = nn.Sequential(
                nn.Linear(node_size * 2, node_size),
                nn.SiLU(),
                nn.Linear(node_size, node_size * 3),
            )
        else:
            self.update_mlp = nn.Sequential(
                nn.Linear(node_size * 2, node_size),
                nn.SiLU(),
                nn.Linear(node_size, node_size * 2),
            )

    def forward(self, node_scalar, node_vector):
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)

        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)

        if not self.last_layer:
            a_vv, a_sv, a_ss = torch.split(
                mlp_output,
                node_vector.shape[-1],
                dim=1,
            )

            delta_v = a_vv.unsqueeze(1) * Uv
            inner_prod = torch.sum(Uv * Vv, dim=1)
            delta_s = a_sv * inner_prod + a_ss

            return node_scalar + delta_s, node_vector + delta_v
        else:
            a_sv, a_ss = torch.split(
                mlp_output,
                node_vector.shape[-1],
                dim=1,
            )

            inner_prod = torch.sum(Uv * Vv, dim=1)
            delta_s = a_sv * inner_prod + a_ss

            return node_scalar + delta_s


def sinc_expansion(edge_dist: torch.Tensor, num_radial: int, cutoff: float):
    """
    Calculate sinc radial basis function:

    sin(n * pi * d / d_cut) / d
    """
    n = torch.arange(num_radial, device=edge_dist.device) + 1
    return torch.sin(edge_dist * n * torch.pi / cutoff) / edge_dist


def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:

    f(d) = 0.5 * (cos(pi * d / d_cut) + 1) for d < d_cut and 0 otherwise
    """
    return torch.where(
        edge_dist < cutoff,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype),
    )
