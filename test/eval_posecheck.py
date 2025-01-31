import os
from rdkit import Chem
import torch
from glob import glob
import pandas as pd
import numpy as np
import contextlib
from tqdm import tqdm
from posecheck import PoseCheck
from rdkit.Chem.QED import qed
import sys
sys.path.append("/root/project/bfn_mol/")
from core.evaluation.utils.sascorer import compute_sa_score
import argparse

pdb_blocks = {}
pdb_proteins = {}
molist = []
show_global = False
pc = PoseCheck()
def find_result_pt_files(directory):
    result_pt_files = glob(os.path.join(directory,"**", '*.sdf'), recursive=True)
    result_pt_files = [file for file in result_pt_files if 'docking_results' not in file]
    return result_pt_files
def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper

def load_results_from_pt(pt_path):
    with open(pt_path, 'rb') as f:
        metrics = torch.load(f)
        if 'all_results' in metrics: # TargetDiff style
            results = metrics['all_results']
        else: # Our style
            results = []
            for graph in metrics:
                if 'mol' not in graph: continue
                data = {'mol': graph.mol, 'ligand_filename': graph.ligand_filename}
                results.append(data)
    return results

def load_mols_from_pt(pt_path):
    results = load_results_from_pt(pt_path)
    molist = []
    for idx, r in enumerate(results):
        try:
            mol = r['mol']
            mol.SetProp('_Name', r['ligand_filename'])
            mol.SetProp('vina_score', str(r['vina']['score_only'][0]['affinity']))
            mol.SetProp('vina_minimize', str(r['vina']['minimize'][0]['affinity']))
            mol.SetProp('vina_dock', str(r['vina']['dock'][0]['affinity']))
        except Exception as e:
            print(e, idx)
            break
        molist.append(mol)
    return molist

def get_hbond(mol, protein_root):
    # redirect stdout to devnull to suppress output
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            try:
                ligand_fn = mol.GetProp('_Name')
                protein_fn = os.path.join(
                    protein_root,
                    os.path.dirname(ligand_fn),
                    os.path.basename(ligand_fn)[:10] + '.pdb'
                )
                pc.load_protein_from_pdb(protein_fn)
                # ref_fn = os.path.join(
                #     protein_root,
                #     ligand_fn
                # )
                # pc.load_ligands_from_sdf(ref_fn)
                pc.load_ligands_from_mols([mol])
                interaction = pc.calculate_interactions()
                interaction.to_csv(f"tmp/{ligand_fn.replace('/', '_')}.csv")
                # Count the occurrences of True in 'HBAcceptor' and 'HBDonor' columns
                df = pd.read_csv(f"tmp/{ligand_fn.replace('/', '_')}.csv")
                # Create a dictionary to map column names to unique ones
                column_mapping = {}
                counter = {}
                for column in df.columns:
                    if column not in column_mapping:
                        column_mapping[column] = column
                        counter[column] = 1
                    else:
                        column_mapping[column] = f"{column}_{counter[column]}"
                        counter[column] += 1

                # Rename the columns with unique names
                df.rename(columns=column_mapping, inplace=True)

                # Count the occurrences of True in 'HBAcceptor' and 'HBDonor' columns
                hb_acceptor_columns = [col for col in column_mapping if col.startswith('HBAcceptor')]
                if len(hb_acceptor_columns) > 0:
                    hb_acceptor_count = sum(df[col].value_counts().get(True, 0) for col in hb_acceptor_columns)
                else:
                    hb_acceptor_count = 0

                hb_donor_columns = [col for col in column_mapping if col.startswith('HBDonor')]
                if len(hb_donor_columns) > 0:
                    hb_donor_count = sum(df[col].value_counts().get(True, 0) for col in hb_donor_columns)
                else:
                    hb_donor_count = 0
                return (hb_acceptor_count, hb_donor_count)

            except Exception as e:
                print(e, ligand_fn)
                return None



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mol_dir', type=str,default='/root/project/bfn_mol/results/denovo/K_32/')
    #parser.add_argument('--protein_root', type=str, default='/root/project/bfn_mol/data/crossdocked_test')
    parser.add_argument('--protein_root', type=str, default='/root/project/bfn_mol/data/test_set')
    args = parser.parse_args()

    mol_dirs = [args.mol_dir]
    ligand_fn = None
    for mol_dir in mol_dirs:
        base_result_path = os.path.join(mol_dir, "saved_data")
        mol_fns = find_result_pt_files(base_result_path)
        # if 'diff' in mol_dir or 'flag' in mol_dir:
        #     mol_fns = sorted(mol_fns, key=lambda x: int(os.path.basename(x).split('.')[0]))
        # else:
        #     mol_fns = sorted(mol_fns, key=lambda x: int(os.path.basename(x).split('.')[0]) % 100)
        vina_scores = []
        vina_mins = []
        vina_docks = []
        clashes = []
        strain_energies = []
        qeds = []
        sas = []
        sizes = []
        hbond_acceptors = []
        hbond_donors = []

        # latest_time = os.path.getctime(f'{mol_dir}/968.sdf')
        for idx_fold, mol_fn in tqdm(enumerate(mol_fns), total=len(mol_fns), desc=os.path.basename(mol_dir)):
            os.makedirs(mol_dir + '/posecheck', exist_ok=True)

            try:
                suppl = Chem.SDMolSupplier(mol_fn, removeHs=False)
                mol = suppl[0]
                if mol is None:
                    continue
                molist.append(mol)

                idx = os.path.basename(mol_fn).split('.')[0]

                if os.path.exists(f'{mol_dir}/posecheck/{idx_fold}_{idx}.sdf'): continue
                if mol is None: continue
            except (OSError, IOError) as e:
                print(f'File error at {mol_fn}: {e}')
                continue
            # 捕获重构错误
            except Exception as e:
                print(e, idx)
                continue

            try:
                # if not mol.HasProp('vina_score'): continue
                smiles = Chem.MolToSmiles(mol)
                if '.' in smiles: continue
                # if mol is None or not mol.HasProp('vina_score'): continue
                
                # record vina affinities
                # vina_scores.append(float(mol.GetProp('vina_score')))
                # if mol.HasProp('vina_minimize'):
                #     vina_mins.append(float(mol.GetProp('vina_minimize')))
                # elif mol.HasProp('vina_min'):
                #     vina_mins.append(float(mol.GetProp('vina_min')))
                # if mol.HasProp('vina_dock'): vina_docks.append(float(mol.GetProp('vina_dock')))
                
                # calculate & record clash and strain
                relative_path = os.path.relpath(mol_fn, base_result_path)
                pdb_file = '/'.join([relative_path.split('/')[0], '_'.join(relative_path.split('/')[1].split('_')[:3])])
                protein_fn = os.path.join(args.protein_root, pdb_file + ".pdb")
                # pdb_file = '/'.join([relative_path.split('/')[0], '_'.join(relative_path.split('/')[1].split('_'))])

                # protein_fn = os.path.join(args.protein_root, pdb_file +"_pocket10" + ".pdb")
                # if mol.GetProp('_Name') != ligand_fn:
                #     ligand_fn = mol.GetProp('_Name')
                #     protein_fn = os.path.join(
                #         args.protein_root,
                #         os.path.dirname(ligand_fn),
                #         os.path.basename(ligand_fn)[:10] + '.pdb'
                #     )
                pc.load_protein_from_pdb(protein_fn)
                pc.load_ligands_from_mols([mol])
                clash = pc.calculate_clashes()[0]
                strain = pc.calculate_strain_energy()[0]
                clashes.append(clash)
                if strain != strain:
                    strain = 1e10
                strain_energies.append(strain)

                qed_val = qed(mol)
                sa_score = compute_sa_score(mol)
                qeds.append(qed_val)
                sas.append(sa_score)
                sizes.append(mol.GetNumAtoms())

                mol.SetProp('clash', str(clash))
                mol.SetProp('strain', str(strain))

                # hb = get_hbond(mol, args.protein_root)
                # if hb is None: continue
                # (hb_acceptor_count, hb_donor_count) = hb
                # mol.SetProp('hb_acceptor', str(hb_acceptor_count))
                # mol.SetProp('hb_donor', str(hb_donor_count))
                # hbond_acceptors.append(hb_acceptor_count)
                # hbond_donors.append(hb_donor_count)

                with Chem.SDWriter(f'{mol_dir}/posecheck/{idx_fold}_{idx}.sdf') as w:
                    w.write(mol)

            except Exception as e:
                print(e, mol_fn)
                continue

        print('file', mol_dir)
        # if len(vina_scores) == 0: continue
        # print('vina_score', np.mean(vina_scores), np.median(vina_scores), np.std(vina_scores))
        # print('vina_min', np.mean(vina_mins), np.median(vina_mins), np.std(vina_mins))
        # if vina_docks: print('vina_dock', np.mean(vina_docks), np.median(vina_docks), np.std(vina_docks))
        if clashes: print('clash', np.mean(clashes), np.median(clashes), np.std(clashes))

        if strain_energies: print('strain', np.quantile(strain_energies, 0.25), np.median(strain_energies), np.quantile(strain_energies, 0.75), np.std(strain_energies))
        print('qed', np.mean(qeds), np.median(qeds), np.std(qeds))
        print('sa', np.mean(sas), np.median(sas), np.std(sas))
        print('size', np.mean(sizes), np.median(sizes), np.std(sizes))
        # print('hbond_acceptors', sum(hbond_acceptors) / len(hbond_acceptors))
        # print('hbond_donors', sum(hbond_donors) / len(hbond_donors))
if __name__ == '__main__':
    main()