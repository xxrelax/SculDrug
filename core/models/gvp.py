import copy
from torch import nn
import torch
from torch_cluster import knn_graph, radius_graph
from torch_geometric.utils import scatter
from math import pi
from core.models.common import MLP, ConcatSquashLinear, GaussianSmearing, compose_context_cross , SinusoidalPosEmb
def compose_context(h_surface, h_ligand_pos, surface_pos, init_ligand_pos, batch_surface, batch_ligand):
    """
    将表面和配体的几何信息组合到一起，生成统一的上下文特征和位置数据。
    """
    batch_all = torch.cat([batch_surface, batch_ligand], dim=0)
    sort_idx = torch.argsort(batch_all, stable=True)  

    mask_ligand = torch.cat([
        torch.zeros(batch_surface.size(0), dtype=torch.bool, device=batch_surface.device),
        torch.ones(batch_ligand.size(0), dtype=torch.bool, device=batch_ligand.device)
    ])[sort_idx]

    h_pos = torch.cat([h_surface, h_ligand_pos], dim=0)[sort_idx]
    pos_all = torch.cat([surface_pos, init_ligand_pos], dim=0)[sort_idx]
    batch_all = batch_all[sort_idx]
    return h_pos, pos_all, mask_ligand, batch_all


class EdgeMapping(nn.Module):
    def __init__(self, edge_channels):
        super().__init__()
        self.nn = nn.Linear(in_features=1, out_features=edge_channels, bias=False)
    
    def forward(self, edge_vector):
        edge_vector = edge_vector / (torch.norm(edge_vector, p=2, dim=1, keepdim=True) + 1e-7)
        expansion = self.nn(edge_vector.unsqueeze(-1))
        flattened_expansion = expansion.flatten(start_dim=1)
        return flattened_expansion



class EdgeProcessor(nn.Module):    
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()   
        self.edge_mlp = nn.Sequential(MLP(in_dim, out_dim, hidden_dim), nn.LayerNorm(out_dim))
    def forward(self, h_i, h_j, edge_attr, edge_mask=None):
        out = torch.cat([h_i, h_j, edge_attr], dim=-1)
        out = self.edge_mlp(out)
        return edge_attr + out


class NodeProcessor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(MLP(in_dim, out_dim, hidden_dim), nn.LayerNorm(out_dim))

    def forward(self, x, edge_index, edge_attr) :
        j = edge_index[1]
        out = scatter(edge_attr, index=j, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=-1)
        out = self.node_mlp(out)
        return x + out
    
class GraphNetsConv(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.edge_processor = EdgeProcessor(node_dim * 2 + edge_dim, node_dim, 2*node_dim)
        self.node_processor = NodeProcessor(node_dim + edge_dim, node_dim, 2*node_dim)
    def forward(self, x, edge_index, edge_attr):
        i = edge_index[0]
        j = edge_index[1]
        edge_attr = self.edge_processor(x[i], x[j], edge_attr)
        x = self.node_processor(x, edge_index, edge_attr)
        return x, edge_attr
    


class Boundary_Awareness_GNN(nn.Module):
    def __init__(self, num_layers, pos_dim=64, time_emb_dim=64, edge_dim=15):
        super().__init__()
        self.num_layers = num_layers
        self.pos_dim = pos_dim
        r_max = 10
        num_r_gaussian = 19
        self.edge_expansion = EdgeMapping(edge_dim)
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian, fixed_offset=False)
        # Position encoders
        self.surface_pos_encoder = nn.Linear(3, pos_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.ligand_pos_encoder = ConcatSquashLinear(3, pos_dim, time_emb_dim)

        self.convs = nn.ModuleList([copy.deepcopy(GraphNetsConv(64, 64)) for _ in range(self.num_layers)])
        self.position_mlp = MLP(pos_dim, 3, pos_dim)
    def filter_edges(self, edge_index, mask_ligand):
        """
        过滤边，只保留从 surface_pos 到 init_ligand_pos 的边。
        
        Args:
            edge_index: 完整的边索引，形状为 (2, num_edges)。
            mask_ligand: 掩码，表示哪些节点是配体节点。

        Returns:
            过滤后的边索引。
        """
        src, dst = edge_index
        valid_mask = (~mask_ligand[src]) & mask_ligand[dst]
        filtered_edge_index = edge_index[:, valid_mask]
        return filtered_edge_index
    def forward(self, surface_pos, init_ligand_pos, batch_surface, batch_ligand, time):
        h_surface_pos = self.surface_pos_encoder(surface_pos)
        h_time = self.time_emb(time.squeeze(-1))
        h_ligand_pos = self.ligand_pos_encoder(h_time, init_ligand_pos)
        h_node, pos_all, mask_ligand, batch_all  = compose_context(h_surface_pos, h_ligand_pos, surface_pos, init_ligand_pos, batch_surface, batch_ligand)
        edge_index = radius_graph(pos_all, 3, batch=batch_all, flow='source_to_target')
        edge_index = self.filter_edges(edge_index, mask_ligand)
        edge_index_src  = edge_index[0]
        edge_index_dst = edge_index[1]
        edge_vector = pos_all[edge_index_src] - pos_all[edge_index_dst]
        edge_vec_feat = self.edge_expansion(edge_vector) 
        edge_dist  = torch.norm(edge_vector, p=2, dim=-1, keepdim=True)
        edge_sca_feat = self.distance_expansion(edge_dist) # 20 
        h_edge = torch.cat([edge_sca_feat, edge_vec_feat], -1) # 64
        for conv in self.convs:
            h_node, h_edge = conv(h_node, edge_index, h_edge)
        pos_ligand = self.position_mlp(h_node)[mask_ligand] +  init_ligand_pos
        return pos_ligand