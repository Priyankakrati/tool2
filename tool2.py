import streamlit as st
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdPartialCharges, Descriptors

from Bio.PDB import PDBParser, NeighborSearch

import py3Dmol
from stmol import showmol

st.set_page_config(layout="wide")

# ================================
# RNA FIELD CONSTRUCTION
# ================================

def classify_atom(atom):
    name = atom.get_name()

    if atom.element == 'O' and 'P' in name:
        return "NEG"
    elif atom.element in ['O', 'N']:
        return "HBOND"
    elif atom.get_parent().get_resname().strip() in ['A', 'G']:
        return "AROMATIC"
    else:
        return "NEUTRAL"


def estimate_charge(atom):
    if atom.element == 'O' and 'P' in atom.get_name():
        return -1.0
    return 0.0


def build_field(pocket_atoms):
    field = []

    for atom in pocket_atoms:
        field.append({
            "coord": atom.coord,
            "type": classify_atom(atom),
            "charge": estimate_charge(atom),
            "vdw": 1.5
        })

    return field


# ================================
# POCKET EXTRACTION
# ================================

def extract_pocket(structure, cutoff=6.0):
    rna_atoms = []

    for model in structure:
        for chain in model:
            for res in chain:
                rna_atoms.extend(list(res.get_atoms()))

    ns = NeighborSearch(rna_atoms)

    pocket = set()
    for atom in rna_atoms:
        neighbors = ns.search(atom.coord, cutoff)
        pocket.update(neighbors)

    return list(pocket)


# ================================
# LIGAND SAMPLING
# ================================

def generate_conformers(mol, n=10):
    mol = Chem.AddHs(mol)
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n)

    for cid in ids:
        AllChem.UFFOptimizeMolecule(mol, confId=cid)

    rdPartialCharges.ComputeGasteigerCharges(mol)
    return mol, ids


# ================================
# PHYSICS TERMS
# ================================

def electrostatic(qi, qj, r):
    return (qi * qj) / (4 * (r + 0.5))


def hbond(dist):
    return np.exp(-((dist - 2.8) ** 2)) if dist < 3.5 else 0


def stacking(dist):
    return 1.0 if 3.3 < dist < 4.5 else 0


def vdw(r, sigma=3.5):
    return (sigma / r)**12 - 2*(sigma / r)**6 if r > 0 else 0


# ================================
# FLEXIBLE MATCHING (KEY NOVEL PART)
# ================================

def score_conformer(mol, conf_id, field):
    conf = mol.GetConformer(conf_id)
    score = 0

    for i, atom in enumerate(mol.GetAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        qi = float(atom.GetDoubleProp("_GasteigerCharge"))

        for f in field:
            r = np.linalg.norm(pos - f["coord"])

            score += electrostatic(qi, f["charge"], r)

            if atom.GetAtomicNum() in [7, 8] and f["type"] == "HBOND":
                score += hbond(r)

            if f["type"] == "AROMATIC":
                score += stacking(r)

            score += vdw(r)

    return score


# ================================
# RNA FLEXIBILITY (LOCAL FIELD VARIATION)
# ================================

def perturb_field(field, noise=0.5):
    new_field = []

    for f in field:
        new_coord = f["coord"] + np.random.normal(0, noise, 3)
        new_field.append({**f, "coord": new_coord})

    return new_field


# ================================
# PROBABILITY
# ================================

def compute_probability(scores):
    scores = np.array(scores)
    weights = np.exp(-scores)
    return float(weights.max() / weights.sum())


# ================================
# SCREENING ENGINE
# ================================

def screen(smiles_dict, field):
    results = []

    for name, smi in smiles_dict.items():

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        mol, conf_ids = generate_conformers(mol)

        ensemble_scores = []

        # RNA flexibility via field perturbation
        for _ in range(3):
            f_field = perturb_field(field)

            for cid in conf_ids:
                s = score_conformer(mol, cid, f_field)
                ensemble_scores.append(s)

        prob = compute_probability(ensemble_scores)

        results.append({
            "Ligand": name,
            "Binding_Prob": round(prob, 3),
            "MW": Descriptors.MolWt(mol)
        })

    return pd.DataFrame(results).sort_values("Binding_Prob", ascending=False)


# ================================
# VISUALIZATION
# ================================

def show_structure(pdb_file):
    with open(pdb_file) as f:
        pdb = f.read()

    view = py3Dmol.view(width=700, height=500)
    view.addModel(pdb, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view


# ================================
# STREAMLIT UI
# ================================

st.title("🧬 RNA Ligand Screening (Novel Physics-Based Engine)")

pdb = st.file_uploader("Upload RNA PDB", type="pdb")

if pdb:
    with open("rna.pdb", "wb") as f:
        f.write(pdb.getbuffer())

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", "rna.pdb")

    pocket_atoms = extract_pocket(structure)
    field = build_field(pocket_atoms)

    st.success(f"Pocket atoms: {len(pocket_atoms)}")

    showmol(show_structure("rna.pdb"))

    txt = st.text_area("SMILES input")

    library = {}
    for i, line in enumerate(txt.splitlines()):
        if line.strip():
            library[f"Mol_{i}"] = line.strip()

    if st.button("Run Screening"):

        df = screen(library, field)

        st.dataframe(df)
        st.bar_chart(df.set_index("Ligand")["Binding_Prob"])
