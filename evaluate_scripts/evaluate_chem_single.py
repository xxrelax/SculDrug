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


def eval_single_mol(mol_path, save_path, pdb_path, center, exhaustiveness):

    mol = Chem.SDMolSupplier(mol_path)[0]
    smiles = Chem.MolToSmiles(mol)
    chem_results = scoring_func.get_chem(mol)

    vina_task = VinaDockingTask.from_generated_mol_eval(
        mol, protein_path=pdb_path, center=center)
    
    score_only_results = vina_task.run(mode='score_only', 
                                        exhaustiveness=exhaustiveness, 
                                        save_dir=save_path)
    minimize_results = vina_task.run(mode='minimize', 
                                        exhaustiveness=exhaustiveness,
                                        save_dir=save_path)
    docking_results = vina_task.run(mode='dock', 
                                    exhaustiveness=exhaustiveness,
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
def evaluate_molecules(result_path, pdb_path, verbose=False, eval_ref=True, exhaustiveness=16, center=None):
    receptor_name = os.path.basename(pdb_path).split('.')[0]
    
    # Set up logger
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not verbose:
        RDLogger.DisableLog('rdApp.*')

    # Initialize counters and result lists
    num_samples = 0
    n_eval_success = 0
    results = []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    # Get list of .sdf files to evaluate
    file_list = sorted([file_name for file_name in os.listdir(result_path) if file_name.endswith('.sdf')])

    for file_name in file_list:
        try:
            dock_result_path = os.path.join(result_path, 'docking_results')
            os.makedirs(dock_result_path, exist_ok=True)

            mol_path = os.path.join(result_path, file_name)
            result = eval_single_mol(mol_path, dock_result_path, pdb_path, center, exhaustiveness)

            n_eval_success += 1
            results.append(result)

        except Exception as e:
            if verbose:
                logger.warning(f'Evaluation failed for {mol_path}: {e}')
            continue

    logger.info(f'Evaluation done! {n_eval_success} samples in total.')

    # Calculate metrics and log the results
    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info(f'QED:   Mean: {np.mean(qed):.3f} Median: {np.median(qed):.3f}')
    logger.info(f'SA:    Mean: {np.mean(sa):.3f} Median: {np.median(sa):.3f}')

    vina_score_only = [r['vina']['score_only']['affinity'] for r in results]
    vina_min = [r['vina']['minimize']['affinity'] for r in results]
    logger.info(f'Vina Score:  Mean: {np.mean(vina_score_only):.3f} Median: {np.median(vina_score_only):.3f}')
    logger.info(f'Vina Min  :  Mean: {np.mean(vina_min):.3f} Median: {np.median(vina_min):.3f}')
    
    vina_dock = [r['vina']['dock']['affinity'] for r in results]
    logger.info(f'Vina Dock :  Mean: {np.mean(vina_dock):.3f} Median: {np.median(vina_dock):.3f}')
    
    # Filter results based on vina docking score
    result_filter = [result for result in results if result['vina']['dock']['affinity'] < 0]
    vina_dock = [r['vina']['dock']['affinity'] for r in result_filter]
    vina_dock_idx = np.argsort(vina_dock)

    # Prepare results for saving
    file_names = [result_filter[i]['ligand_filename'] for i in vina_dock_idx]
    chem_results = [result_filter[i]['chem_results'] for i in vina_dock_idx]
    vina_results = [result_filter[i]['vina']['dock']['affinity'] for i in vina_dock_idx]
    vina_min_results = [result_filter[i]['vina']['minimize']['affinity'] for i in vina_dock_idx]
    vina_score_only_results = [result_filter[i]['vina']['score_only']['affinity'] for i in vina_dock_idx]
    smiles = [result_filter[i]['smiles'] for i in vina_dock_idx]

    # Create a DataFrame to store the results
    df = pd.DataFrame({
        'file_names': ['docking_results/docked_' + os.path.basename(filename) for filename in file_names],
        'smiles': smiles,
        'vina_dock_result': vina_results,
        'vina_min_result': vina_min_results,
        'vina_score_result': vina_score_only_results,
        'qed': [chem['qed'] for chem in chem_results],
        'sa': [chem['sa'] for chem in chem_results],
        'logp': [chem['logp'] for chem in chem_results],
        'lipinski': [chem['lipinski'] for chem in chem_results],
    })

    # Save results to CSV and PyTorch file
    df.to_csv(os.path.join(result_path, 'molecule_properties.csv'), index=False)
    torch.save(results, os.path.join(result_path, 'chem_eval_results.pt'))

    # If evaluating reference ligand
    if eval_ref:
        ref_mol_path = os.path.join(os.path.dirname(pdb_path), os.path.basename(result_path) + '.sdf')
        ref_result = eval_single_mol(ref_mol_path, dock_result_path, pdb_path, center, exhaustiveness)
        torch.save(ref_result, os.path.join(result_path, 'chem_reference_results.pt'))
        logger.info('Reference ligand evaluation done!')

        # Append reference ligand results to DataFrame and save
        df = pd.read_csv(os.path.join(result_path, 'molecule_properties.csv'))
        new_row = pd.DataFrame([{
            'file_names': 'reference',
            'smiles': ref_result['smiles'],
            'vina_dock_result': ref_result['vina']['dock']['affinity'],
            'vina_min_result': ref_result['vina']['minimize']['affinity'],
            'vina_score_result': ref_result['vina']['score_only']['affinity'],
            'qed': ref_result['chem_results']['qed'],
            'sa': ref_result['chem_results']['sa'],
            'logp': ref_result['chem_results']['logp'],
            'lipinski': ref_result['chem_results']['lipinski']
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(os.path.join(result_path, 'molecule_properties.csv'), index=False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--result_path', type=str, default='/root/project/bfn_mol/results/denovo/bfnmol/saved_data/ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0')
    parser.add_argument('--pdb_path', type=str, default='/root/project/bfn_mol/data/test_set/ABL2_HUMAN_274_551_0/4xli_B_rec.pdb')
    parser.add_argument('--eval_ref', type=bool, default=True)
    parser.add_argument('--exhaustiveness', type=int, default=1)
    parser.add_argument('--center', type=float, nargs=3, default=None,
                        help='Center of the pocket bounding box, in format x,y,z') # [4.35 , 3.75, 3.16] for adrb1  [1.30, -3.75, -1.90] for drd3
    args = parser.parse_args()
    evaluate_molecules(args.result_path, args.pdb_path, args.exhaustiveness, args.eval_ref, args.verbose)
