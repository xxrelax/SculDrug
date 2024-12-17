import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum

from core.models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product

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
    def aggregate_to_virtual_center_by_batch(self, h_protein, group_indices, virtual_mask, batch):
        """
        将同一批次中的同一簇的真实原子表征聚合到该簇的虚拟中心原子上，使用平均聚合方式。

        Args:
            h_protein: 蛋白质特征张量 (N_protein, feature_dim)
            group_indices: 每个原子的簇索引，用于标识各个簇
            virtual_mask: 虚拟原子的掩码，用于区分真实和虚拟原子
            batch: 批次索引，用于区分不同批次的原子 (N_protein,)

        Returns:
            updated_h_protein: 更新后的蛋白质特征张量，簇内真实原子特征聚合到虚拟中心原子上
        """

        # 初始化更新后的特征张量
        updated_h_protein = h_protein.clone()

        # 获取所有批次中的唯一簇索引组合
        unique_batches = torch.unique(batch)

        for b in unique_batches:
            # 获取当前批次的原子索引
            batch_mask = (batch == b)
            batch_group_indices = group_indices[batch_mask]
            batch_virtual_mask = virtual_mask[batch_mask]
            batch_h_protein = h_protein[batch_mask]

            # 获取当前批次的唯一簇索引
            valid_clusters = torch.unique(batch_group_indices)
            valid_clusters = valid_clusters[valid_clusters != -1]

            for cluster in valid_clusters:
                # 获取该簇中的原子索引
                cluster_indices = (batch_group_indices == cluster).nonzero(as_tuple=True)[0]

                virtual_center_idx = cluster_indices[batch_virtual_mask[cluster_indices]][0]

                # 获取簇内真实原子的索引
                real_atom_indices = cluster_indices[~batch_virtual_mask[cluster_indices]]
                cluster_features_mean = batch_h_protein[real_atom_indices].mean(dim=0)
                updated_h_protein[batch_mask][virtual_center_idx] += cluster_features_mean

        return updated_h_protein

    def forward(self, h, x, mask_ligand, batch, virtual_mask, group_indices, return_all=False, fix_x=False):

        all_x = [x]
        all_h = [h]

        for b_idx in range(self.num_blocks):
            h = self.aggregate_to_virtual_center_by_batch(h, group_indices, virtual_mask, batch)
            #global attention
            global_edge_index = self._connect_within_batch(x, virtual_mask, batch, mask_ligand)
            src, dst = global_edge_index
            edge_type = self._build_edge_type(global_edge_index, mask_ligand)
            dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
            dist_feat = self.distance_expansion(dist)
            logits = self.edge_pred_layer(dist_feat)
            e_w = torch.sigmoid(logits)
            for l_idx, layer in enumerate(self.global_block):
                h, x = layer(h, x, edge_type, global_edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)
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
            edge_index_2p7 = radius_graph(x, r=1.35, batch=batch, flow='source_to_target')
            edge_index_3p4 = radius_graph(x, r=1.7, batch=batch, flow='source_to_target')
            edge_index_4p9 = radius_graph(x, r=2.5, batch=batch, flow='source_to_target')
            edge_index_4p9 = remove_subset_edges(edge_index_4p9, edge_index_3p4)
            edge_index_3p4 = remove_subset_edges(edge_index_3p4, edge_index_2p7)
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
        return outputs

