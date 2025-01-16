import copy
import numpy as np
import torch

from time import time
from torch.profiler import profile, record_function, ProfilerActivity

import pytorch_lightning as pl

from torch_scatter import scatter_mean, scatter_sum

from core.config.config import Config
from core.models.bfn4sbdd import BFN4SBDDScoreModel

import core.evaluation.utils.atom_num as atom_num
import core.utils.transforms as trans

from core.utils.train import get_optimizer, get_scheduler

def build_local_coordinate_system(atom_positions, batch_indices, offset):
    """
    针对不同 batch 的分子构建局部坐标系。
    
    Args:
        atom_positions: 原子的坐标张量，shape=(N, 3)，包含所有分子的原子。
        batch_indices: 每个原子所属的 batch，shape=(N,)。
        
    Returns:
        transform_matrices: 每个 batch 的局部坐标系变换矩阵，shape=(B, 3, 3)。
        offset: 每个 batch 的局部坐标系原点（质心），shape=(B, 3)。
    """
    # Step 1: 计算每个 batch 的质心
    batch_size = batch_indices.max() + 1
    expanded_offsets = offset[batch_indices]
    distances = torch.norm(atom_positions - expanded_offsets, dim=1) 
    sorted_indices = torch.argsort(distances)  # 按距离排序
    sorted_batch_indices = batch_indices[sorted_indices]  # 对应 batch 排序

    # 利用分组方式找到每个 batch 中的前两个最近原子
    unique_batches, inverse_indices, counts = torch.unique(sorted_batch_indices, return_inverse=True, return_counts=True)
    cumsum_counts = torch.cumsum(counts, dim=0)
    starts = torch.cat([torch.tensor([0], device=cumsum_counts.device), cumsum_counts[:-1]])  # 每个 batch 的起点索引

    # 获取前两个原子索引
    nearest_indices = torch.stack([
        sorted_indices[starts + 1],  # 最近的第一个原子
        sorted_indices[starts + 2]  # 最近的第二个原子
    ], dim=1)

    # Step 4: 构建局部坐标系
    pos_A = offset  # 质心
    pos_B = atom_positions[nearest_indices[:, 0]]  # 最近的第一个原子
    pos_C = atom_positions[nearest_indices[:, 1]]  # 最近的第二个原子

    x_axis = (pos_B - pos_A) / torch.norm(pos_B - pos_A, dim=1, keepdim=True)
    temp_vector = (pos_C - pos_A)
    z_axis = torch.cross(x_axis, temp_vector, dim=1)
    z_axis /= torch.norm(z_axis, dim=1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis, dim=1)

    transform_matrices = torch.stack([x_axis, y_axis, z_axis], dim=1)  # shape=(B, 3, 3)

    return transform_matrices

def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode="protein"):
    if mode == "none":
        offset = 0.0
        pass
    elif mode == "protein":
        offset = scatter_mean(protein_pos, batch_protein, dim=0) # tensor([[14.0434, 17.8929, 51.6457]
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


class SBDDTrainLoop(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.dynamics = BFN4SBDDScoreModel(**self.cfg.dynamics.todict())
        # [ time, h_t, pos_t, edge_index]
        self.train_losses = []
        self.save_hyperparameters(self.cfg.todict())
        self.time_records = np.zeros(6)
        self.log_time = False

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        t1 = time()
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, virtual_mask, group_indices_list, surface_pos, batch_surface = (
            batch.protein_pos,
            batch.protein_atom_feature.float(),
            batch.protein_element_batch,
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
            batch.virtual_mask,
            batch.group_indices_list,
            batch.surface_pos,
            batch.surface_pos_batch,
        )
        # get the data from the batch
        # batch is a data object
        # protein_pos: [N_pro,3]
        # protein_v: [N_pro,27]
        # batch_protein: [N_pro]
        # ligand_pos: [N_lig,3]
        # ligand_v: [N_lig,13]
        # protein_element_batch: [N_protein]

        t2 = time()
        #sum(batch.protein_pos)
#tensor([206720.2969, 263387.4688, 760219.3125], device='cuda:3')
        #
        with torch.no_grad():
            # add noise to protein_pos
            protein_noise = torch.randn_like(protein_pos) * self.cfg.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            # random rotation as data aug
            if self.cfg.train.random_rot:
                M = np.random.randn(3, 3)
                Q, __ = np.linalg.qr(M)
                Q = torch.from_numpy(Q.astype(np.float32)).to(ligand_pos.device)
                gt_protein_pos = gt_protein_pos @ Q
                ligand_pos = ligand_pos @ Q

        num_graphs = batch_protein.max().item() + 1
        # !!!!!
        
        gt_protein_pos, ligand_pos, offset = center_pos(
            gt_protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            mode=self.cfg.dynamics.center_pos_mode,
        )  # TODO: ugly
        # tensor([[ 6.3049, 20.0927, 47.8768],
        surface_pos = surface_pos - offset[batch_surface] # tensor([[-7.7385,  2.1997, -3.7689],
#         sum(transform_matrix)
# tensor([[-7.5741e+00, -1.0321e+01, -2.9235e+01],
#         [ 1.9786e+00, -2.0772e-01,  3.1158e-02],
#         [-2.3701e-02, -1.6483e+00,  1.1887e+00]], device='cuda:3')
        transform_matrix = build_local_coordinate_system(gt_protein_pos, batch_protein, offset) # tensor([[[-0.2852, -0.3520, -0.8915], 
        #         sum(transform_matrix[batch_protein])
        # tensor([[ -3484.0942,  -4747.6875, -13447.3213],
        #         [   600.6031,   -738.0328,    -42.0148],
        #         [  -635.1982,   -581.0387,    415.1340]], device='cuda:3')
#         sum(gt_protein_pos)
# tensor([-0.0220,  0.0368,  0.1392], device='cuda:3')
#sum(gt_protein_pos)
# tensor([-0.0224,  0.0320,  0.1251], device='cuda:3')
#
        #sum(batch_protein)
# tensor(228160, device='cuda:3')
#sum(offset)
#tensor([ 449.3897,  572.5743, 1652.6633], device='cuda:3')
        gt_protein_pos = torch.matmul(gt_protein_pos.unsqueeze(1), transform_matrix[batch_protein]).squeeze(1) # tensor([[-0.1629,  3.7592,  3.3262],  # tensor([[ 2.5848,  3.3384,  2.3394],
        ligand_pos = torch.matmul(ligand_pos.unsqueeze(1), transform_matrix[batch_ligand]).squeeze(1) # tensor([[ 0.5836,  0.0562,  0.2090], # tensor([[ 0.5389,  0.2922,  0.1487], #tensor([[ 0.4619, -0.2212,  0.3162],
        surface_pos = torch.matmul(surface_pos.unsqueeze(1), transform_matrix[batch_surface]).squeeze(1) # tensor([[-1.1245,  0.3584,  8.4099], tensor([[-1.4966,  4.9915,  7.3254],
        # gt_protein_pos = gt_protein_pos / self.cfg.data.normalizer

        t3 = time()
        t = torch.rand(
            [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device
        ).index_select(
            0, batch_ligand
        )  # different t for different molecules.

        if not self.cfg.dynamics.use_discrete_t and not self.cfg.dynamics.destination_prediction:
            # t = torch.randint(0, 999, [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device).index_select(0, batch_ligand) #different t for different molecules.
            # t = t / 1000.0
            # else:
            t = torch.clamp(t, min=self.dynamics.t_min)  # clamp t to [t_min,1]

        t4 = time()
        c_loss, d_loss, discretised_loss, g_loss = self.dynamics.loss_one_step(
            t,
            protein_pos=gt_protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            ligand_pos=ligand_pos,
            ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            virtual_mask = virtual_mask, 
            group_indices_list =  group_indices_list,
            surface_pos = surface_pos, 
            batch_surface = batch_surface
        )

        # here the discretised_loss is close for current version.

        loss = torch.mean(c_loss + self.cfg.train.v_loss_weight * d_loss + discretised_loss + 0.2*g_loss)

        t5 = time()
        self.log_dict(
            {
                'lr': self.get_last_lr(),
                'loss': loss.item(),
            },
            on_step=True,
            prog_bar=True,
            batch_size=self.cfg.train.batch_size,
        )
        self.log_dict(
            {
                'loss_pos': c_loss.mean().item(), 
                'loss_type': d_loss.mean().item(),
                'loss_c_ratio': c_loss.mean().item() / loss.item(),
                "loss_g": g_loss.mean().item(),
            },
            on_step=True,
            prog_bar=False,
            batch_size=self.cfg.train.batch_size,
        )

        # check if loss is finite, skip update if not
        if not torch.isfinite(loss):
            return None
        self.train_losses.append(loss.clone().detach().cpu())

        t0 = time()

        if self.log_time:
            self.time_records = np.vstack((self.time_records, [t0, t1, t2, t3, t4, t5]))
            print(f'step total time: {self.time_records[-1, 0] - self.time_records[-1, 1]}, batch size: {num_graphs}')
            print(f'\tpl call & data access: {self.time_records[-1, 1] - self.time_records[-2, 0]}')
            print(f'\tunwrap data: {self.time_records[-1, 2] - self.time_records[-1, 1]}')
            print(f'\tadd noise & center pos: {self.time_records[-1, 3] - self.time_records[-1, 2]}')
            print(f'\tsample t: {self.time_records[-1, 4] - self.time_records[-1, 3]}')
            print(f'\tget loss: {self.time_records[-1, 5] - self.time_records[-1, 4]}')
            print(f'\tlogging: {self.time_records[-1, 0] - self.time_records[-1, 5]}')
        return loss

    def validation_step(self, batch, batch_idx):
        # out_data_list = self.shared_sampling_step(batch, batch_idx, sample_num_atoms='prior', desc=f'Val')
        # return out_data_list
        pass


    def test_step(self, batch, batch_idx):
        # TODO change order, samples of the same pocket should be together, reduce protein loading
        out_data_list = []
        n_samples = self.cfg.evaluation.num_samples
        for _ in range(n_samples):
            batch_output = self.shared_sampling_step(batch, batch_idx, sample_num_atoms=self.cfg.evaluation.sample_num_atoms, 
                                                     desc=f'Test-{_}/{n_samples}')
            # for idx, item in enumerate(batch_output):
            out_data_list.append(batch_output)
                
        out_data_list_reorder = []
        for i in range(len(out_data_list[0])):
            for j in range(len(out_data_list)):
                out_data_list_reorder.append(out_data_list[j][i])
        return out_data_list_reorder

    def shared_sampling_step(self, batch, batch_idx, sample_num_atoms, desc=''):
        # here we need to sample the molecules in the validation step
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, virtual_mask, group_indices_list, surface_pos, batch_surface = (
            batch.protein_pos,
            batch.protein_atom_feature.float(),
            batch.protein_element_batch,
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
            batch.virtual_mask,
            batch.group_indices_list,
            batch.surface_pos, 
            batch.surface_pos_batch
        )
        num_graphs = batch_protein.max().item() + 1  # B
        n_nodes = batch_ligand.size(0)  # N_lig
        # assert num_graphs == len(batch), f"num_graphs: {num_graphs} != len(batch): {len(batch)}"

        # move protein center to origin & ligand correspondingly
        protein_pos, ligand_pos, offset = center_pos(
            protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            mode=self.cfg.dynamics.center_pos_mode,
        )  # TODO: ugly
        
        surface_pos = surface_pos - offset[batch_surface]
        transform_matrix = build_local_coordinate_system(protein_pos, batch_protein, offset)
        protein_pos = torch.matmul(protein_pos.unsqueeze(1), transform_matrix[batch_protein]).squeeze(1)
        ligand_pos = torch.matmul(ligand_pos.unsqueeze(1), transform_matrix[batch_ligand]).squeeze(1)
        surface_pos = torch.matmul(surface_pos.unsqueeze(1), transform_matrix[batch_surface]).squeeze(1)
        # determine the number of atoms in the ligand
        if sample_num_atoms == 'prior':
            ligand_num_atoms = []
            for data_id in range(len(batch)):
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy() * self.cfg.data.normalizer_dict.pos)
                ligand_num_atoms.append(atom_num.sample_atom_num(pocket_size).astype(int))
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(ligand_pos.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=ligand_pos.device)
        elif sample_num_atoms == 'ref':
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).to(ligand_pos.device)
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")
        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=ligand_pos.device), 
            ligand_num_atoms.cumsum(dim=0)
        ])

        # TODO reuse for visualization and test
        # forward pass to get the ligand sample
        theta_chain, sample_chain, y_chain = self.dynamics.sample(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            virtual_mask=virtual_mask, 
            group_indices_list=group_indices_list,
            surface_pos=surface_pos, 
            batch_surface=batch_surface,
            # n_nodes=n_nodes,
            sample_steps=self.cfg.evaluation.sample_steps,
            n_nodes=num_graphs,
            # ligand_pos=ligand_pos,  # for debug only
            desc=desc,
        )

        # restore ligand to original position
        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        pred_pos, one_hot = final[0], final[1]
        inverse_transform_matrix = transform_matrix.transpose(-1, -2)
        pred_pos = torch.matmul(pred_pos.unsqueeze(1), inverse_transform_matrix[batch_ligand]).squeeze(1)
        pred_pos = pred_pos + offset[batch_ligand]
        # along with normalizer
        pred_pos = pred_pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=ligand_pos.device
        )
        out_batch = copy.deepcopy(batch)
        out_batch.protein_pos = out_batch.protein_pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=ligand_pos.device
        )

        pred_v = one_hot.argmax(dim=-1)
        # TODO: ugly, should be done in metrics.py (but needs a way to make it compatible with pyg batch)
        pred_atom_type = trans.get_atomic_number_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[int]

        # for visualization
        atom_type = [trans.MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in pred_atom_type]  # List[int]
        atom_type = torch.tensor(atom_type, dtype=torch.long, device=ligand_pos.device)  # [N_lig]

        # for reconstruction
        pred_aromatic = trans.is_aromatic_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[bool]

        # print('[DEBUG]', num_graphs, len(ligand_cum_atoms))
        
        # add necessary dict to new pyg batch
        out_batch.x, out_batch.pos = atom_type, pred_pos
        out_batch.atom_type = torch.tensor(pred_atom_type, dtype=torch.long, device=ligand_pos.device)
        out_batch.is_aromatic = torch.tensor(pred_aromatic, dtype=torch.long, device=ligand_pos.device)
        # out_batch.mol = results

        _slice_dict = {
            "x": ligand_cum_atoms,
            "pos": ligand_cum_atoms,
            "atom_type": ligand_cum_atoms,
            "is_aromatic": ligand_cum_atoms,
            # "mol": out_batch._slice_dict["ligand_filename"],
        }
        _inc_dict = {
            "x": out_batch._inc_dict["ligand_element"], # [0] * B, TODO: figure out what this is
            "pos": out_batch._inc_dict["ligand_pos"],
            "atom_type": out_batch._inc_dict["ligand_element"],
            "is_aromatic": out_batch._inc_dict["ligand_element"],
            # "mol": out_batch._inc_dict["ligand_filename"],
        }
        out_batch._inc_dict.update(_inc_dict)
        out_batch._slice_dict.update(_slice_dict)
        out_data_list = out_batch.to_data_list()
        return out_data_list

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        print(f"epoch_loss: {epoch_loss}")
        self.log(
            "epoch_loss",
            epoch_loss,
            batch_size=self.cfg.train.batch_size,
        )
        self.train_losses = []

    def configure_optimizers(self):
        self.optim = get_optimizer(self.cfg.train.optimizer, self)
        self.scheduler, self.get_last_lr = get_scheduler(self.cfg.train, self.optim)

        return {
            'optimizer': self.optim, 
            'lr_scheduler': self.scheduler,
        }
