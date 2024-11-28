import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.nn import radius_graph, knn_graph
import torch_scatter
from core.models.common import ConcatSquashLinear, GaussianSmearing, MLP, NONLINEARITIES
from core.models.common import SinusoidalPosEmb
from torch_geometric.nn import GCNConv

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
    # GNN的等变基层，假设已实现
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn=nn.ReLU(), norm=True):
        super().__init__()
        # 假设该层包含一些GNN等变层的定义
        self.layer = GCNConv(hidden_dim, hidden_dim)  # 示例使用 GCN
    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x, edge_index)
    
    
# class GNN(nn.Module):
#     def __init__(self, num_layers, hidden_dim, time_emb_dim):
#         super().__init__()
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
        
#         # Position encoders
#         self.surface_pos_encoder = nn.Linear(3, hidden_dim)
#         self.time_emb = nn.Sequential(
#             SinusoidalPosEmb(time_emb_dim),
#             nn.Linear(time_emb_dim, time_emb_dim * 4),
#             nn.GELU(),
#             nn.Linear(time_emb_dim * 4, time_emb_dim)
#         )
#         self.ligand_pos_encoder = ConcatSquashLinear(3, hidden_dim, time_emb_dim)
        
#         # GNN network layers
#         self.net = self._build_network()
#         self.position_mlp = nn.Linear(hidden_dim, 3)
#     def _build_network(self):
#         # 构建 GNN 网络层
#         layers = []
#         for _ in range(self.num_layers):
#             layer = EnBaseLayer(self.hidden_dim, edge_feat_dim=32, num_r_gaussian=10)
#             layers.append(layer)
#         return nn.ModuleList(layers)

#     def aggregate_to_virtual_nodes(self, h_surface, virtual_edge_index):
#         """
#         聚合所有与虚拟节点相连的 surface 节点的信息。

#         Args:
#             h_surface (torch.Tensor): 表示 surface 节点特征的张量，形状为 (N_surface, feature_dim)。
#             virtual_edge_index (torch.Tensor): 表示 surface 节点和虚拟节点之间边连接的索引，形状为 (2, num_edges)。
#                 virtual_edge_index[0] 是 surface 节点的索引，virtual_edge_index[1] 是对应的虚拟节点索引。

#         Returns:
#             virtual_node_features (torch.Tensor): 每个虚拟节点的聚合特征，形状为 (N_virtual, feature_dim)。
#         """
#         num_virtual_nodes = virtual_edge_index[1].max().item() + 1

#         virtual_node_features = torch_scatter.scatter_mean(
#             h_surface[virtual_edge_index[0]],   
#             virtual_edge_index[1],              
#             dim=0,
#             dim_size=num_virtual_nodes          
#         )

#         return virtual_node_features
#     def _connect_within_batch(self, x, virtual_mask, batch, mask_ligand):
#         """
#         在同一个批次内连接原子和虚拟原子，生成全局的边索引

#         Args:
#             x (torch.Tensor): 所有节点特征张量，形状为 (N, feature_dim)。
#             virtual_mask (torch.Tensor): 布尔掩码，标记哪些节点是虚拟原子。
#             batch (torch.Tensor): 每个节点所在批次的索引，形状为 (N,)。
#             mask_ligand (torch.Tensor): 布尔掩码，标记哪些节点是配体原子。

#         Returns:
#             torch.Tensor: 全局边索引 `global_edge_index`，形状为 (2, num_edges)。
#         """
#         device = x.device

#         non_protein_mask = virtual_mask | mask_ligand
#         non_protein_indices = torch.where(non_protein_mask)[0]

        
#         max_batch = batch.max().item() + 1

#         all_indices = non_protein_indices.unsqueeze(0).repeat(non_protein_indices.size(0), 1)
#         src_indices = all_indices.flatten()  # 源节点索引
#         dst_indices = all_indices.T.flatten()  # 目标节点索引

#         # 去除对角线元素（自连接）
#         mask = src_indices != dst_indices
#         src_indices, dst_indices = src_indices[mask], dst_indices[mask]

#         src_batch = batch[src_indices]
#         dst_batch = batch[dst_indices]
#         batch_mask = src_batch == dst_batch
#         src_indices, dst_indices = src_indices[batch_mask], dst_indices[batch_mask]
#         global_edge_index = torch.stack([src_indices, dst_indices], dim=0).to(device)

#         return global_edge_index
#     def compose_context(self, h_surface, h_ligand_pos, batch_protein, batch_ligand, virtual_mask):
       
#         h_all = torch.cat([h_surface, h_ligand_pos], dim=0)  # (N_surface + N_ligand, feature_dim)

#         batch = torch.cat([batch_protein, batch_ligand], dim=0)  # (N_surface + N_ligand,)

#         surface_virtual_mask = torch.cat([virtual_mask, torch.zeros_like(batch_ligand, dtype=torch.bool)], dim=0)  # 虚拟节点掩码
#         mask_ligand = torch.cat([torch.zeros_like(batch_protein, dtype=torch.bool), torch.ones_like(batch_ligand, dtype=torch.bool)], dim=0)  # 配体掩码

#         sorted_indices = batch.argsort()  # 获取排序索引
#         batch = batch[sorted_indices]  # 按批次索引排序
#         h_all = h_all[sorted_indices]
#         surface_virtual_mask = surface_virtual_mask[sorted_indices]
#         mask_ligand = mask_ligand[sorted_indices]

#         return h_all, batch, surface_virtual_mask, mask_ligand
#     def forward(self, surface_pos, init_ligand_pos, surface_group_indices, surface_virtual_mask, batch_surface, batch_ligand, time):
#         # 1. 特征编码
#         h_surface = self.surface_pos_encoder(surface_pos)
#         h_time = self.time_emb(time)
#         h_ligand_pos = self.ligand_pos_encoder(h_time, init_ligand_pos)

#         # 2. 连接虚拟节点并聚合
#         virtual_edge_index = connect_virtual_edge(h_surface, surface_group_indices, surface_virtual_mask, batch_surface)
#         virtual_node_features = self.aggregate_to_virtual_nodes(h_surface, virtual_edge_index)

#         # 3. 合并上下文
#         h_all, batch, surface_virtual_mask, mask_ligand = self.compose_context(
#             virtual_node_features, h_ligand_pos, batch_protein=batch_surface, batch_ligand=batch_ligand, virtual_mask=surface_virtual_mask
#         )

#         # 4. 构建全局边索引
#         global_edge_index = self._connect_within_batch(h_all, surface_virtual_mask, batch, mask_ligand)

#         # 5. 应用 GNN 层
#         for layer in self.net:
#             h_all = layer(h_all, global_edge_index)
        
#         predicted_positions = self.position_mlp(h_all[mask_ligand])
#         return predicted_positions


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

    def forward(self, surface_pos, init_ligand_pos, batch_surface, batch_ligand, time):
        h_surface = self.surface_pos_encoder(surface_pos)
        h_time = self.time_emb(time.squeeze(-1))
        h_ligand_pos = self.ligand_pos_encoder(h_time, init_ligand_pos)

        combined_pos = torch.cat([surface_pos, init_ligand_pos], dim=0)
        combined_batch = torch.cat([batch_surface, batch_ligand], dim=0)
        edge_index = knn_graph(combined_pos, k=30, batch=combined_batch)


        h_combined = torch.cat([h_surface, h_ligand_pos], dim=0)
        for layer in self.net:
            h_combined = layer(h_combined, edge_index)

        mask_ligand = torch.cat(
            [torch.zeros_like(batch_surface, dtype=torch.bool), torch.ones_like(batch_ligand, dtype=torch.bool)]
        )
        predicted_positions = self.position_mlp(h_combined[mask_ligand])
        return predicted_positions

