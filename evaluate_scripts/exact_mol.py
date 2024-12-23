import os
import torch
import numpy as np
from rdkit import Chem
from collections import defaultdict
import argparse

def save_mol(mol, filename):
    """Save RDKit molecule object as .sdf file"""
    writer = Chem.SDWriter(filename)
    writer.write(mol)
    writer.close()

def process_molecule_data(save_dir, generated_file):
    # Load generated data
    results = torch.load(generated_file)

    # Group results by the first part of the ligand_filename
    grouped_results = defaultdict(list)
    for result in results:
        group_name = result['ligand_filename'].split('/')[0]  # Extract directory name
        grouped_results[group_name].append(result)

    # Save each group of results
    for group_name, group_results in grouped_results.items():
        group_save_dir = os.path.join(save_dir, group_name)
        os.makedirs(group_save_dir, exist_ok=True)
        
        # Save each molecule's data in the group
        for count, result in enumerate(group_results, 1):
            # Generate file path for saving
            pdb_filename = '_'.join(result['ligand_filename'].split('_')[:7])
            entry = [pdb_filename + '.pdb', result['ligand_filename']]
            data = {
                'pos': np.array(result['pred_pos']),
                'atom': np.array(result['pred_v']),
                'entry': entry
            }
            
            # Create directory for saving the result
            group_result_dir = os.path.join(save_dir, result['ligand_filename'][:-4])
            os.makedirs(group_result_dir, exist_ok=True)
            
            # Save the .pt file
            torch.save(data, os.path.join(group_result_dir, f'sample_{count:04d}.pt'))
            
            # Save the .sdf file dynamically
            save_mol(result['mol'], os.path.join(group_result_dir, f'sample_{count:04d}.sdf'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process molecule data and save it.")
    parser.add_argument('--save_dir', type=str, default='/root/project/bfn_mol/results/denovo/all_mult_new/saved_data',
                        help="Directory where results will be saved (default: '/root/project/bfn_mol/results/denovo/all_mult_new/saved_data')")
    parser.add_argument('--generated_file', type=str, 
                        default="/root/project/bfn_mol/logs/root_bfn_sbdd/all_mult_new/0/test_outputs_v3/20241220-092053/generated.pt",
                        help="Path to the generated .pt file (default: '/root/project/bfn_mol/logs/root_bfn_sbdd/add_cluster_vert/2/test_outputs_v9/20241206-223130/generated.pt')")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    process_molecule_data(args.save_dir, args.generated_file)
