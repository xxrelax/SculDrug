import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.nn import radius_graph, knn_graph
import torch_scatter
from core.models.common import ConcatSquashLinear, GaussianSmearing, MLP, NONLINEARITIES
from core.models.common import SinusoidalPosEmb
from torch_geometric.nn import GCNConv

from core.models.gvp import compose_context

class EnBaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn='silu', norm=False):
        super().__init__()
        self.r_min = 0.
        self.r_max = 10.
        self.hidden_dim = hidden_dim
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        if num_r_gaussian > 1:
            self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)
        self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + num_r_gaussian, hidden_dim, hidden_dim,
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True)
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        if self.update_x:
            # self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)
            x_mlp = [nn.Linear(hidden_dim, hidden_dim), NONLINEARITIES[act_fn]]
            layer = nn.Linear(hidden_dim, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            x_mlp.append(layer)
            x_mlp.append(nn.Tanh())
            self.x_mlp = nn.Sequential(*x_mlp)

        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)

    def forward(self, h, x, edge_index, mask_ligand, edge_attr=None):
        src, dst = edge_index
        hi, hj = h[dst], h[src]
        # \phi_e in Eq(3)
        rel_x = x[dst] - x[src]
        d_sq = torch.sum(rel_x ** 2, -1, keepdim=True)
        if self.num_r_gaussian > 1:
            d_feat = self.distance_expansion(torch.sqrt(d_sq + 1e-8))
        else:
            d_feat = d_sq
        if edge_attr is not None:
            edge_feat = torch.cat([d_feat, edge_attr], -1)
        else:
            edge_feat = d_sq

        mij = self.edge_mlp(torch.cat([hi, hj, edge_feat], -1))
        eij = self.edge_inf(mij)
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])

        # h update in Eq(6)
        h = h + self.node_mlp(torch.cat([mi, h], -1))
        if self.update_x:
            # x update in Eq(4)
            xi, xj = x[dst], x[src]
            # (xi - xj) / (\|xi - xj\| + C) to make it more stable
            delta_x = scatter_sum((xi - xj) / (torch.sqrt(d_sq + 1e-8) + 1) * self.x_mlp(mij), dst, dim=0)
            x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated

        return h, x

class BaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn=nn.ReLU(), norm=True):
        super().__init__()
        self.layer = GCNConv(hidden_dim, hidden_dim)
    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x, edge_index)
    


class BaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim=None, num_r_gaussian=None, update_x=True, act_fn=nn.ReLU(), norm=True):
        super().__init__()
        self.layer = GCNConv(hidden_dim, hidden_dim)  # 示例使用 GCN
    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x, edge_index)


class GNN(nn.Module):
    def __init__(self, num_layers, hidden_dim, time_emb_dim):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Position encoders
        self.surface_pos_encoder = nn.Linear(3, hidden_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.ligand_pos_encoder = ConcatSquashLinear(3, hidden_dim, time_emb_dim)

        # GNN network layers
        self.net = self._build_network()
        self.position_mlp = nn.Linear(hidden_dim, 3)

    def _build_network(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(BaseLayer(self.hidden_dim))
        return nn.ModuleList(layers)

    def filter_edges(self, edge_index, mask_ligand):
        src, dst = edge_index
        valid_mask = (~mask_ligand[src]) & mask_ligand[dst]
        filtered_edge_index = edge_index[:, valid_mask]
        return filtered_edge_index
    def forward(self, surface_pos, init_ligand_pos, batch_surface, batch_ligand, time):
        h_surface_pos = self.surface_pos_encoder(surface_pos)
        h_time = self.time_emb(time.squeeze(-1))
        h_ligand_pos = self.ligand_pos_encoder(h_time, init_ligand_pos)
        h_node, pos_all, mask_ligand, batch_all  = compose_context(h_surface_pos, h_ligand_pos, surface_pos, init_ligand_pos, batch_surface, batch_ligand)
        edge_index = radius_graph(pos_all, 3.5, batch=batch_all, flow='source_to_target')
        edge_index = self.filter_edges(edge_index, mask_ligand)
        for layer in self.net:
            h_combined = layer(h_node, edge_index)

        mask_ligand = torch.cat(
            [torch.zeros_like(batch_surface, dtype=torch.bool), torch.ones_like(batch_ligand, dtype=torch.bool)]
        )
        predicted_positions = self.position_mlp(h_combined[mask_ligand])
        return predicted_positions

