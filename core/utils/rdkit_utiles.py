from rdkit.Chem import AllChem as Chem
from collections import Counter

def ring_type_from_mol(mol):
    ring_info = mol.GetRingInfo()
    ring_type = [len(r) for r in ring_info.AtomRings()]
    return ring_type

def clean_frags(mol, threshold=10, filter_ring=False):
    mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    if mol.GetNumAtoms() < threshold:
        return Chem.MolFromSmiles('C.C')
    
    if filter_ring:
        ring_type = Counter(ring_type_from_mol(mol))
        if 4 in ring_type:
            mol = Chem.MolFromSmiles('C.C')
        if 3 in ring_type and ring_type[3]>1:
            mol = Chem.MolFromSmiles('C.C')
        for num in ring_type.keys():
            if num >=7:
                mol = Chem.MolFromSmiles('C.C')
    return mol
def evaluate_validity(mol, threshold=None, threshold_ratio=0.8):
    '''
    clean the frags away from backbones and determin the validity
    the molecule size > tshd will be perserved.

    threshold >= 0: tshd = threshold
    threshold = None: no cleaning
    threshold < 0: tshd = num_atom * 0.8
    '''

    if threshold is not None:

        threshold = int(threshold)
        if threshold < 0:
            threshold = int(threshold_ratio * mol.GetNumAtoms())

        mol = clean_frags(mol, 
                          threshold=threshold, 
                          filter_ring=False)
    
    smiles = Chem.MolToSmiles(mol)
    
    if '.' in smiles:
        return mol, False
    else:    
        return mol, True