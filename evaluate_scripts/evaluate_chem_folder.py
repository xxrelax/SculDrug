import os
import argparse
import subprocess
from pytorch_lightning import seed_everything
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count
import sys
sys.path.append('/root/project/bfn_mol')
from evaluate_scripts.evaluate_chem_single import evaluate_molecules

def get_all_deepest_subfolders(base_path):

    deepest_subfolders = []
    for root, dirs, files in os.walk(base_path):
        if not dirs:  
            deepest_subfolders.append(root)
    return deepest_subfolders

def run_evaluation(args):
    result_path, base_result_path, base_pdb_path, exhaustiveness, eval_ref, verbose = args
    try:
        relative_path = os.path.relpath(result_path, base_result_path)
        pdb_file = '/'.join([relative_path.split('/')[0], '_'.join(relative_path.split('/')[1].split('_')[:3])])
        pdb_sub_path = os.path.join(base_pdb_path, pdb_file + ".pdb")

        if os.path.exists(pdb_sub_path):
            print(f"Processing {result_path} with PDB {pdb_sub_path}")
            evaluate_molecules(result_path, pdb_sub_path, verbose = verbose, eval_ref =eval_ref, exhaustiveness= exhaustiveness)
            # cmd = [
            #     "python", "./evaluate_scripts/evaluate_chem_single.py",
            #     "--result_path", result_path,
            #     "--pdb_path", pdb_sub_path,
            #     "--exhaustiveness", str(exhaustiveness),
            #     "--eval_ref", str(eval_ref),
            #     "--verbose", str(verbose)
            # ]
            # subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error processing {result_path}: {e}")

def main(base_result_path, base_pdb_path, exhaustiveness, eval_ref, verbose):
    deepest_subfolders = get_all_deepest_subfolders(base_result_path)

    nthreads = 32
    print("Number of CPU cores:", nthreads)

    args_list = []
    for result_path in deepest_subfolders:
        if 'docking_results' in result_path:
            result_path = os.path.dirname(result_path)
        args_list.append((result_path, base_result_path, base_pdb_path, exhaustiveness, eval_ref, verbose))
    with Pool(processes=nthreads) as pool:
        for _ in tqdm(pool.imap(run_evaluation, args_list), total=len(args_list)):
            pass
    # for result_path in tqdm(deepest_subfolders):
    #     if 'docking_results' in result_path:
    #         result_path = os.path.dirname(result_path)
        
    #     args = (result_path, base_result_path, base_pdb_path, exhaustiveness, eval_ref, verbose)
        
    #     run_evaluation(args)
    # print('evaluation done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_result_path', type=str, default='/root/project/bfn_mol/baseline/results/graphbp/saved_data', help="Base result path to traverse")
    parser.add_argument('--base_pdb_path', type=str, default='/root/project/bfn_mol/data/test_set', help="Base PDB path for constructing pdb_path")
    parser.add_argument('--exhaustiveness', type=int, default=1, help="Exhaustiveness parameter for Vina docking")
    parser.add_argument('--eval_ref', type=bool, default=True, help="Whether to evaluate the reference ligand")
    parser.add_argument('--verbose', type=eval, default=False, help="Verbose output")
    parser.add_argument("--seed", type=int, default=722, help="Random seed")
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)
    main(args.base_result_path, args.base_pdb_path, args.exhaustiveness, args.eval_ref, args.verbose)
