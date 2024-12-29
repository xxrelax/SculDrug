import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum
import torch_scatter
from time import time
from core.models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product

def multi_radius_graph(x, batch, radii):
    """
    根据多个半径阈值从节点坐标和批次信息中生成多级别的边集合。
    只调用一次 radius_graph，然后根据距离进行分级筛选。

    Args:
        x (Tensor): 节点坐标, [N, D]
        batch (LongTensor): 节点的批次信息, [N]
        radii (list or tuple): 递增顺序的半径列表 (例如 [1.35, 1.7, 2.5])

    Returns:
        edges_list (list of Tensors): 对应 radii 阈值范围划分的边集列表
    """

    # 确保 radii 已经排序
    radii = sorted(radii)
    r_max = radii[-1]

    # 使用最大半径一次性构建边
    edge_index_full = radius_graph(x, r=r_max, batch=batch, flow='source_to_target')

    # 计算每条边的距离
    row, col = edge_index_full
    # 假设 x 为 (N, D)
    # 计算欧式距离的平方（避免调用 sqrt 提高性能）
    diff = x[row] - x[col]
    dist_sq = (diff * diff).sum(dim=-1)  # dist^2

    # 将半径也转为平方方便比较
    radii_sq = [r*r for r in radii]

    edges_list = []
    # 上一个半径区间的上界
    prev_r_sq = 0.0  
    for r_sq in radii_sq:
        # 选取 dist_sq 在 (prev_r_sq, r_sq] 区间内的边
        # 如果希望第一个区间包括从0到r1的所有边，那么 prev_r_sq可设0
        mask = (dist_sq <= r_sq) & (dist_sq > prev_r_sq)
        edges_list.append(edge_index_full[:, mask])
        prev_r_sq = r_sq

    return edges_list
def remove_subset_edges(main_edge_index, subset_edge_index):
# 将边对转换为集合进行操作
    main_edges = set(map(tuple, main_edge_index.t().tolist()))
    subset_edges = set(map(tuple, subset_edge_index.t().tolist()))

    # 从 main_edges 中移除 subset_edges
    filtered_edges = main_edges - subset_edges

    # 转换回 tensor 格式
    filtered_edge_index = torch.tensor(list(filtered_edges), dtype=torch.long, device=main_edge_index.device).t()
    return filtered_edge_index
class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        # compute k
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        # compute v
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output


class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * self.edge_feat_dim,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * self.edge_feat_dim,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, knn=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False, name='unio2net'):
        super().__init__()
        self.name = name
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.knn = knn
        self.ew_net_type = ew_net_type  # [r, m, none]

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block, self.inter_cluster_block, self.global_block = self._build_share_blocks()
        self.aggre_mlp = MLP(128,128,256)
    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        inter_cluster_block = []
        global_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, 7, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)

        # for l_idx in range(2):
        #     layer = AttentionLayer_inter_cluster(
        #         self.hidden_dim, self.n_heads, self.num_r_gaussian, edge_feat_dim=0, act_fn=self.act_fn,
        #         norm=self.norm,
        #         num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
        #         ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        #     )
        #     inter_cluster_block.append(layer)

        for l_idx in range(2):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, 4, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            global_block.append(layer)
        
        return nn.ModuleList(base_block), nn.ModuleList(inter_cluster_block), nn.ModuleList(global_block)

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            edge_index = knn_graph(x, k=self.knn, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.knn, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index
    def _connect_all_edge(self,x,batch):
        edge_index = []
        for b in torch.unique(batch):
            batch_nodes = (batch == b).nonzero(as_tuple=True)[0]
            edges = torch.combinations(batch_nodes, r=2).t()
            edge_index.append(edges)
        edge_index = torch.cat(edge_index, dim=1)
        return edge_index
    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type
    
    def _connect_virtual_edge(self, x, group_indices, virtual_mask, batch):
        virtual_indices = torch.where(virtual_mask)[0]


        cluster_ids = group_indices[virtual_indices]
        batch_ids = batch[virtual_indices]

        # 找到所有与虚拟节点相同 cluster 和 batch 的节点索引
        source_edges = torch.where((group_indices.unsqueeze(1) == cluster_ids) & 
                                (batch.unsqueeze(1) == batch_ids))

        target_edges = virtual_indices[source_edges[1]]
        source_edges = source_edges[0]

        edge_index = torch.stack([source_edges, target_edges], dim=0).to(x.device)
        
        return edge_index
    def _connect_virtual_edge(self, x, group_indices, virtual_mask, batch):
        edges = []

        virtual_indices = torch.where(virtual_mask)[0]

        for virtual_idx in virtual_indices:
            cluster_id = group_indices[virtual_idx].item()
            batch_id = batch[virtual_idx].item()

            nodes_in_cluster = torch.where((group_indices == cluster_id) & (batch == batch_id))[0]
            
            if nodes_in_cluster.numel() > 0:
                source_edges = nodes_in_cluster
                target_edges = virtual_idx.expand(nodes_in_cluster.size(0))
                cluster_edges = torch.stack([source_edges, target_edges], dim=0)
                edges.append(cluster_edges)

        if edges:
            edge_index = torch.cat(edges, dim=1).to(x.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)  # 返回空的 edge_index

        return edge_index
    def _connect_within_batch(self, x, virtual_mask, batch, mask_ligand):
        """
        建立同一批次中分子与分子、虚拟原子与虚拟原子、分子与虚拟原子之间的连接。
        """

        device = x.device
        non_protein_mask = virtual_mask | mask_ligand
        non_protein_indices = torch.where(non_protein_mask)[0]
        unique_batches = batch[non_protein_indices].unique()
        edge_index = []
        for b in unique_batches:
            nodes_in_batch = non_protein_indices[batch[non_protein_indices] == b]
            if nodes_in_batch.size(0) > 1:
                edges = torch.combinations(nodes_in_batch, r=2)
                edge_index.append(edges)
                edge_index.append(edges.flip(dims=[1]))
        edge_index = torch.cat(edge_index, dim=0).t().to(device)
        return edge_index
    # def aggregate_to_virtual_center_by_batch(self, h_protein, group_indices, virtual_mask, batch):
    #     """
    #     将同一批次中的同一簇的真实原子表征聚合到该簇的虚拟中心原子上，使用平均聚合方式。

    #     Args:
    #         h_protein: 蛋白质特征张量 (N_protein, feature_dim)
    #         group_indices: 每个原子的簇索引，用于标识各个簇
    #         virtual_mask: 虚拟原子的掩码，用于区分真实和虚拟原子
    #         batch: 批次索引，用于区分不同批次的原子 (N_protein,)

    #     Returns:
    #         updated_h_protein: 更新后的蛋白质特征张量，簇内真实原子特征聚合到虚拟中心原子上
    #     """

    #     # 初始化更新后的特征张量
    #     updated_h_protein = h_protein.clone()

    #     # 获取所有批次中的唯一簇索引组合
    #     unique_batches = torch.unique(batch)

    #     for b in unique_batches:
    #         # 获取当前批次的原子索引
    #         batch_mask = (batch == b)
    #         batch_group_indices = group_indices[batch_mask]
    #         batch_virtual_mask = virtual_mask[batch_mask]
    #         batch_h_protein = h_protein[batch_mask]

    #         # 获取当前批次的唯一簇索引
    #         valid_clusters = torch.unique(batch_group_indices)
    #         valid_clusters = valid_clusters[valid_clusters != -1]

    #         for cluster in valid_clusters:
    #             # 获取该簇中的原子索引
    #             cluster_indices = (batch_group_indices == cluster).nonzero(as_tuple=True)[0]

    #             virtual_center_idx = cluster_indices[batch_virtual_mask[cluster_indices]][0]

    #             # 获取簇内真实原子的索引
    #             real_atom_indices = cluster_indices[~batch_virtual_mask[cluster_indices]]
    #             cluster_features_mean = batch_h_protein[real_atom_indices].mean(dim=0)
    #             updated_h_protein[virtual_center_idx] += cluster_features_mean

    #     return updated_h_protein
    
    def _connect_within_batch_optimized(self, x, virtual_mask, batch, mask_ligand):
        """
        优化后的函数：建立同一批次中分子与分子、虚拟原子与虚拟原子、分子与虚拟原子之间的连接。
        
        所有计算均在 GPU (CUDA) 上进行。
        
        Args:
            x: 输入特征张量 (在 GPU 上)
            virtual_mask: 虚拟原子的掩码 (在 GPU 上)
            batch: 批次索引，用于区分不同批次的原子 (N_protein,) (在 GPU 上)
            mask_ligand: 配体掩码，用于区分配体原子 (在 GPU 上)
        
        Returns:
            edge_index: 边索引张量，形状为 (2, E)，在 GPU 上
        """
        device = x.device
        # 确保所有输入张量都在 GPU 上
        # 假设在函数外部已经保证 x, virtual_mask, batch, mask_ligand 均在 GPU 上
        # 如果不确定，可使用以下语句（根据需要取消注释）：
        # virtual_mask = virtual_mask.to(device)
        # mask_ligand = mask_ligand.to(device)
        # batch = batch.to(device)
        
        non_protein_mask = virtual_mask | mask_ligand
        non_protein_indices = torch.where(non_protein_mask)[0]  # 位于 GPU 上
        
        # 如果非蛋白原子数量小于等于1，则无需建立边
        if non_protein_indices.size(0) <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        
        batch_non_protein = batch[non_protein_indices]  # 在 GPU 上
        N = batch_non_protein.size(0)
        
        # 在 GPU 上生成所有上三角（不含对角线）的索引对
        idx_i, idx_j = torch.triu_indices(N, N, offset=1, device=device)  # 直接在 GPU 上生成
        # 筛选出同一批次内的原子对
        same_batch_mask = (batch_non_protein[idx_i] == batch_non_protein[idx_j])
        
        # 获取符合条件的原子索引对
        selected_idx_i = idx_i[same_batch_mask]
        selected_idx_j = idx_j[same_batch_mask]
        
        edges = torch.stack([
            non_protein_indices[selected_idx_i], 
            non_protein_indices[selected_idx_j]
        ], dim=0)
        
        # 生成双向边
        edges = torch.cat([edges, edges.flip(dims=[0])], dim=1)
        return edges
    def aggregate_to_virtual_center_by_batch_optimized(self, h_protein, group_indices, virtual_mask, batch):
        # 优化版本函数（需要确保 group_indices >= 0 且从0开始计数）
        valid_mask = group_indices != -1
        if valid_mask.sum() == 0:
            return h_protein.clone()

        combined = batch * (group_indices.max()+1) + group_indices
        valid_combined = combined[valid_mask]
        real_mask = valid_mask & ~virtual_mask

        unique_groups, inverse_indices = torch.unique(valid_combined, sorted=True, return_inverse=True)
        num_unique_groups = unique_groups.size(0)

        real_indices = torch.nonzero(real_mask, as_tuple=False).squeeze(1)
        real_combined = combined[real_indices]
        real_inverse = torch.searchsorted(unique_groups, real_combined)
        real_features = h_protein[real_indices]

        sum_features = torch_scatter.scatter_add(real_features, real_inverse, dim=0, dim_size=num_unique_groups)
        counts = torch_scatter.scatter_add(torch.ones_like(real_inverse, dtype=torch.float32), real_inverse, dim=0, dim_size=num_unique_groups)
        group_mean = sum_features / counts.unsqueeze(1).clamp(min=1.0)
        group_mean = self.aggre_mlp(group_mean)

        virtual_mask_valid = valid_mask & virtual_mask
        if virtual_mask_valid.sum() == 0:
            return h_protein.clone()

        virtual_indices = torch.nonzero(virtual_mask_valid, as_tuple=False).squeeze(1)
        virtual_combined = combined[virtual_indices]
        virtual_inverse = torch.searchsorted(unique_groups, virtual_combined)

        updated_h_protein = h_protein.clone()
        updated_h_protein[virtual_indices] += group_mean[virtual_inverse]

        return updated_h_protein
    def forward(self, h, x, mask_ligand, batch, virtual_mask, group_indices, return_all=False, fix_x=False):

        all_x = [x]
        all_h = [h]
        # t1 = time()
        for b_idx in range(self.num_blocks):
            h = self.aggregate_to_virtual_center_by_batch_optimized(h, group_indices, virtual_mask, batch)
            #global attention
            global_edge_index = self._connect_within_batch_optimized(x, virtual_mask, batch, mask_ligand)
            src, dst = global_edge_index
            edge_type = self._build_edge_type(global_edge_index, mask_ligand)
            dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
            dist_feat = self.distance_expansion(dist)
            logits = self.edge_pred_layer(dist_feat)
            e_w = torch.sigmoid(logits)
            for l_idx, layer in enumerate(self.global_block):
                h, x = layer(h, x, edge_type, global_edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)
            h = h[~virtual_mask]
            x = x[~virtual_mask]
            batch = batch[~virtual_mask]
            mask_ligand = mask_ligand[~virtual_mask]
            # mask_ligand = mask_ligand[~virtual_mask]
            # radius_list = torch.tensor([2.7, 3.4, 4.9], device="cuda")
            # distance_matrix = torch.cdist(x, x)
            # radius_mask = distance_matrix.unsqueeze(-1) <= radius_list
            # edge_indices = (radius_mask).nonzero(as_tuple=False)
            # edge_index = edge_indices[:, :2].T
            # edge_type_r = edge_indices[:, 2]
            # self_loop_mask = edge_index[0] != edge_index[1]
            # edge_index = edge_index[:, self_loop_mask]
            # edge_type_r = edge_type_r[self_loop_mask]
            # edge_type_atom = F.one_hot(edge_type_r, num_classes=3)
            
            # 多级边
            # edge_index_2p7 = radius_graph(x, r=1.35, batch=batch, flow='source_to_target')
            # edge_index_3p4 = radius_graph(x, r=1.7, batch=batch, flow='source_to_target')
            # edge_index_4p9 = radius_graph(x, r=2.5, batch=batch, flow='source_to_target')
            # edge_index_4p9 = remove_subset_edges(edge_index_4p9, edge_index_3p4)
            # edge_index_3p4 = remove_subset_edges(edge_index_3p4, edge_index_2p7)
            radii = [1.35, 1.7, 2.5]
            edge_index_levels = multi_radius_graph(x, batch, radii)
            edge_index_2p7 = edge_index_levels[0]  # 对应半径1.35内的边
            edge_index_3p4 = edge_index_levels[1]  # 对应半径1.35~1.7区间的边
            edge_index_4p9 = edge_index_levels[2]  # 对应半径1.7~2.5区间的边
            edge_type_2p7 = torch.full((edge_index_2p7.size(1),), 0, dtype=torch.long, device=edge_index_2p7.device)  # 类型 0
            edge_type_3p4 = torch.full((edge_index_3p4.size(1),), 1, dtype=torch.long, device=edge_index_3p4.device)  # 类型 1
            edge_type_4p9 = torch.full((edge_index_4p9.size(1),), 2, dtype=torch.long, device=edge_index_4p9.device)  # 类型 2
            edge_index = torch.cat([edge_index_2p7, edge_index_3p4, edge_index_4p9], dim=-1)
            edge_type_atom = torch.cat([edge_type_2p7, edge_type_3p4, edge_type_4p9], dim=-1)
            edge_type_atom = F.one_hot(edge_type_atom, num_classes=3)
            
            # edge_index = self._connect_edge(x, mask_ligand, batch)
            src, dst = edge_index
            edge_type_mol = self._build_edge_type(edge_index, mask_ligand)
            edge_type = edge_type_mol
            edge_type = torch.cat([edge_type_atom, edge_type_mol], dim=-1)            
  
            # edge_type_type = self._build_edge_type(edge_index, mask_ligand)
            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            for l_idx, layer in enumerate(self.base_block):
                h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)
            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        # t2 = time()
        # print(f"block time: {t2-t1}")
        return outputs

