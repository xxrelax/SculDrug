import argparse
from functools import partial
import multiprocessing
import os, sys

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(repo_dir)

from core.evaluation.utils import scoring_func

from core.evaluation.docking_vina import VinaDockingTask
from core.utils import misc
import pandas as pd

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


def eval_single_mol(mol,mol_path, save_path):
    smiles = Chem.MolToSmiles(mol)
    chem_results = scoring_func.get_chem(mol)
    
    vina_task = VinaDockingTask.from_generated_mol(
        mol, mol_path, protein_root=args.pdb_path, center=args.center)
    
    score_only_results = vina_task.run(mode='score_only', 
                                        exhaustiveness=args.exhaustiveness, 
                                        save_dir=save_path)
    minimize_results = vina_task.run(mode='minimize', 
                                        exhaustiveness=args.exhaustiveness,
                                        save_dir=save_path)
    docking_results = vina_task.run(mode='dock', 
                                    exhaustiveness=args.exhaustiveness,
                                    save_dir=save_path)
    
    vina_results = {
        'score_only': score_only_results,
        'minimize': minimize_results,
        'dock': docking_results
    }

    return {
            'mol': mol,
            'smiles': smiles,
            'ligand_filename': mol_path,
            'chem_results': chem_results,
            'vina': vina_results,
            'num_atoms': mol.GetNumAtoms()
        }

def eval_and_save(gene_data, result_path, logger, verbose=False):
    try:
        dock_result_path = os.path.join(result_path, 'docking_results')
        os.makedirs(dock_result_path, exist_ok=True)

        result = eval_single_mol(gene_data["mol"], gene_data["ligand_filename"], dock_result_path)
        return result

    except Exception as e:
        if verbose:
            logger.warning(f"Evaluation failed for {gene_data['ligand_filename']}: {str(e)}")
        return None  
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_path', type=str, default="./logs/root_bfn_sbdd/add_cluster_vert/0/test_outputs_v2/20241203-113241/generated.pt")
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--result_path', type=str, default='./results/add_cluster_vert/0/')
    parser.add_argument('--pdb_path', type=str, default='/root/project/bfn_mol/data/test_set')
    parser.add_argument('--eval_ref', type=bool, default=False)
    parser.add_argument('--exhaustiveness', type=int, default=48)
    parser.add_argument('--center', type=float, nargs=3, default=None,
                        help='Center of the pocket bounding box, in format x,y,z') # [4.35 , 3.75, 3.16] for adrb1  [1.30, -3.75, -1.90] for drd3
    args = parser.parse_args()

    receptor_name = args.pdb_path.split('/')[-1].split('.')
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    num_samples = 0
    n_eval_success = 0
    results = []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    gene_list = torch.load(args.generate_path)
    manager = multiprocessing.Manager()
    results = manager.list()
    n_eval_success = manager.Value('i', 0)
    eval_task = partial(eval_and_save, result_path=result_path, logger=logger, verbose=False)
    with multiprocessing.Pool(processes=args.exhaustiveness) as pool:
        eval_results = pool.map(eval_task, gene_list)
        for result in eval_results:
            if result is not None:
                results.append(result)
                n_eval_success.value += 1
    n_eval_success = n_eval_success.value
    logger.info(f'Evaluate done! {n_eval_success} samples in total.')

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))

    vina_score_only = [r['vina']['score_only']['affinity'] for r in results]
    vina_min = [r['vina']['minimize']['affinity'] for r in results]
    logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
    logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
    vina_dock = [r['vina']['dock']['affinity'] for r in results]
    logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))
    
    result_filter = [result for result in results if result['vina']['dock']['affinity'] < 0]
    vina_dock = [r['vina']['dock']['affinity'] for r in result_filter]
    vina_dock_idx = np.argsort(vina_dock)

    file_names = [result_filter[i]['ligand_filename'] for i in vina_dock_idx]
    chem_results = [result_filter[i]['chem_results'] for i in vina_dock_idx]
    vina_results = [result_filter[i]['vina']['dock']['affinity'] for i in vina_dock_idx]
    vina_min_results = [result_filter[i]['vina']['minimize']['affinity'] for i in vina_dock_idx]
    vina_score_only_results = [result_filter[i]['vina']['score_only']['affinity'] for i in vina_dock_idx]

    smiles = [result_filter[i]['smiles'] for i in vina_dock_idx]

    df = pd.DataFrame({'file_names': ['docking_results/docked_' + os.path.split(filename)[1] for filename in file_names],
                       'smiles': smiles,
                       'vina_dock_result': vina_results,
                       'vina_min_result': vina_min_results,
                       'vina_score_result': vina_score_only_results,
                       'qed': [chem['qed'] for chem in chem_results], 'sa': [chem['sa'] for chem in chem_results],
                       'logp': [chem['logp'] for chem in chem_results], 'lipinski': [chem['lipinski'] for chem in chem_results],}, 
                       )

    
    df.to_csv(os.path.join(result_path, 'molecule_properties.csv'), index=False)
    torch.save(results, os.path.join(result_path, 'chem_eval_results.pt'))

    if args.eval_ref:
        sdf_files = []

        # 遍历目录及其子目录
        for root, dirs, files in os.walk(args.pdb_path):
            for file in files:
                if file.endswith('.sdf'):  # 检查文件是否以 .sdf 结尾
                    # 获取文件的相对路径，去掉根目录部分
                    relative_path = os.path.relpath(os.path.join(root, file), args.pdb_path)
                    sdf_files.append(relative_path)
        results = []
        for sdf in sdf_files:
            ref_mol_path = os.path.join(args.pdb_path, sdf)
            mol = Chem.SDMolSupplier(ref_mol_path)[0]
            ref_result = eval_single_mol(mol,sdf, result_path)
            n_eval_success += 1
            results.append(ref_result)
        torch.save(results, os.path.join(result_path, 'chem_reference_results.pt'))
        logger.info('Reference ligand evaluation done!')
        file_names = [result['ligand_filename'] for result in results]
        df = pd.DataFrame({
            'file_names': ['docking_results/docked_' + os.path.split(filename)[1] for filename in file_names],
            'smiles': [result['smiles'] for result in results],
            'vina_dock_result': [result['vina']['dock']['affinity'] for result in results],
            'vina_min_result': [result['vina']['minimize']['affinity'] for result in results],
            'vina_score_result': [result['vina']['score_only']['affinity'] for result in results],
            'qed': [result['chem_results']['qed'] for result in results],
            'sa': [result['chem_results']['sa'] for result in results],
            'logp': [result['chem_results']['logp'] for result in results],
            'lipinski': [result['chem_results']['lipinski'] for result in results]
        })
        df.to_csv(os.path.join(result_path, 'molecule_properties_ref.csv'), index=False)
