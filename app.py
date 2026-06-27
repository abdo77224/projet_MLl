import streamlit as st
import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import GraphDescriptors
from rdkit.Chem.EState import EState_VSA

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction de Solubilité",
    page_icon="🧪",
    layout="centered"
)

# ── Chargement du modèle ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model               = joblib.load('streamlit_model/model.pkl')
    selected_feat_names = joblib.load('streamlit_model/features.pkl')
    return model, selected_feat_names

model, selected_feature_names = load_model()


# ── Calcul des descripteurs ───────────────────────────────────────────────────
def compute_all_descriptors(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc = {}

    # Physicochimiques
    desc['MolWt']              = Descriptors.MolWt(mol)
    desc['HeavyAtomMolWt']     = Descriptors.HeavyAtomMolWt(mol)
    desc['ExactMolWt']         = Descriptors.ExactMolWt(mol)
    desc['MolLogP']            = Descriptors.MolLogP(mol)
    desc['MolMR']              = Descriptors.MolMR(mol)
    desc['TPSA']               = Descriptors.TPSA(mol)
    desc['LabuteASA']          = Descriptors.LabuteASA(mol)
    desc['HeavyAtomCount']     = mol.GetNumHeavyAtoms()
    desc['QED']                = QED.qed(mol)
    desc['FractionCSP3']       = rdMolDescriptors.CalcFractionCSP3(mol)

    # Comptages
    desc['NumHDonors']          = rdMolDescriptors.CalcNumHBD(mol)
    desc['NumHAcceptors']       = rdMolDescriptors.CalcNumHBA(mol)
    desc['NumRotatableBonds']   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    desc['NumHeteroatoms']      = rdMolDescriptors.CalcNumHeteroatoms(mol)
    desc['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
    desc['NHOHCount']           = Descriptors.NHOHCount(mol)
    desc['NOCount']             = Descriptors.NOCount(mol)
    desc['NumRadicalElectrons'] = Descriptors.NumRadicalElectrons(mol)

    # Cycles
    desc['NumAromaticRings']         = rdMolDescriptors.CalcNumAromaticRings(mol)
    desc['NumSaturatedRings']        = rdMolDescriptors.CalcNumSaturatedRings(mol)
    desc['NumAliphaticRings']        = rdMolDescriptors.CalcNumAliphaticRings(mol)
    desc['RingCount']                = rdMolDescriptors.CalcNumRings(mol)
    desc['NumAromaticHeterocycles']  = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    desc['NumAromaticCarbocycles']   = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
    desc['NumSaturatedHeterocycles'] = rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)
    desc['NumSaturatedCarbocycles']  = rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)
    desc['NumAliphaticHeterocycles'] = rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
    desc['NumAliphaticCarbocycles']  = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
    desc['NumSpiroAtoms']            = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    desc['NumBridgeheadAtoms']       = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    desc['NumAmideBonds']            = rdMolDescriptors.CalcNumAmideBonds(mol)
    # ✅ Bug corrigé
    desc['NumStereocenters']         = len(
        Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    )

    # Chi
    desc['Chi0']  = Descriptors.Chi0(mol)
    desc['Chi1']  = Descriptors.Chi1(mol)
    desc['Chi0v'] = Descriptors.Chi0v(mol)
    desc['Chi1v'] = Descriptors.Chi1v(mol)
    desc['Chi2v'] = Descriptors.Chi2v(mol)
    desc['Chi3v'] = Descriptors.Chi3v(mol)
    desc['Chi4v'] = Descriptors.Chi4v(mol)
    desc['Chi0n'] = Descriptors.Chi0n(mol)
    desc['Chi1n'] = Descriptors.Chi1n(mol)
    desc['Chi2n'] = Descriptors.Chi2n(mol)
    desc['Chi3n'] = Descriptors.Chi3n(mol)
    desc['Chi4n'] = Descriptors.Chi4n(mol)

    # Kappa
    desc['Kappa1']        = Descriptors.Kappa1(mol)
    desc['Kappa2']        = Descriptors.Kappa2(mol)
    desc['Kappa3']        = Descriptors.Kappa3(mol)
    desc['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)

    # EState
    desc['MaxAbsEStateIndex'] = Descriptors.MaxAbsEStateIndex(mol)
    desc['MaxEStateIndex']    = Descriptors.MaxEStateIndex(mol)
    desc['MinAbsEStateIndex'] = Descriptors.MinAbsEStateIndex(mol)
    desc['MinEStateIndex']    = Descriptors.MinEStateIndex(mol)

    # PEOE_VSA
    for i in range(1, 15):
        fn = getattr(Descriptors, f'PEOE_VSA{i}', None)
        if fn:
            desc[f'PEOE_VSA{i}'] = fn(mol)

    # SMR_VSA
    for i in range(1, 11):
        fn = getattr(Descriptors, f'SMR_VSA{i}', None)
        if fn:
            desc[f'SMR_VSA{i}'] = fn(mol)

    # SlogP_VSA
    for i in range(1, 13):
        fn = getattr(Descriptors, f'SlogP_VSA{i}', None)
        if fn:
            desc[f'SlogP_VSA{i}'] = fn(mol)

    # EState_VSA
    try:
        for i, v in enumerate(EState_VSA.EState_VSA_(mol)):
            desc[f'EState_VSA{i+1}'] = v
    except Exception:
        pass

    # Topologiques
    try:
        desc['BalabanJ'] = GraphDescriptors.BalabanJ(mol)
    except Exception:
        desc['BalabanJ'] = 0.0
    desc['BertzCT'] = GraphDescriptors.BertzCT(mol)
    try:
        desc['Ipc'] = Descriptors.Ipc(mol, avg=True)
    except Exception:
        desc['Ipc'] = 0.0

    # Features dérivées
    mw = desc['MolWt']
    desc['TPSA_per_MolWt'] = desc['TPSA']    / mw if mw > 0 else 0.0
    desc['LogP_per_MolWt'] = desc['MolLogP'] / mw if mw > 0 else 0.0

    # Morgan Fingerprints 64 bits
    fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=64)
    for i in range(64):
        desc[f'fp_{i}'] = int(fp[i])

    return desc


# ── Prédiction ────────────────────────────────────────────────────────────────
def predict(smiles: str):
    desc = compute_all_descriptors(smiles)
    if desc is None:
        return None, None

    X = pd.DataFrame([desc])
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # Ajouter colonnes manquantes + ordre exact
    for col in selected_feature_names:
        if col not in X.columns:
            X[col] = 0.0

    X_sel = X[selected_feature_names]
    logS  = float(model.predict(X_sel)[0])
    return logS, desc


def interpret_logS(logS: float):
    if logS > -1:
        return "🟢 Très soluble",     "success"
    elif logS > -2:
        return "🟡 Soluble",          "warning"
    elif logS > -4:
        return "🟠 Peu soluble",      "warning"
    else:
        return "🔴 Très peu soluble", "error"


# ── Interface ─────────────────────────────────────────────────────────────────
st.title("🧪 Prédiction de Solubilité Moléculaire")
st.divider()

# ── Exemples rapides ──────────────────────────────────────────────────────────
st.markdown("#### 💡 Molécules exemples")
examples = {
    "Aspirine"   : "CC(=O)Oc1ccccc1C(=O)O",
    "Caféine"    : "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "Glucose"    : "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "Ibuprofène" : "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Benzène"    : "c1ccccc1",
}

cols = st.columns(len(examples))
for col, (name, smi) in zip(cols, examples.items()):
    if col.button(name, use_container_width=True):
        st.session_state['smiles_input'] = smi

# ── Saisie SMILES ─────────────────────────────────────────────────────────────
st.markdown("#### ✏️ Entrez un SMILES")
smiles_input = st.text_input(
    label="SMILES",
    value=st.session_state.get('smiles_input', ''),
    placeholder="Ex: CC(=O)Oc1ccccc1C(=O)O",
    label_visibility="collapsed"
)

if st.button("🔮 Prédire la Solubilité", type="primary", use_container_width=True):

    if not smiles_input.strip():
        st.warning("⚠️ Veuillez entrer un SMILES.")

    elif Chem.MolFromSmiles(smiles_input) is None:
        st.error("❌ SMILES invalide. Vérifiez la structure.")

    else:
        with st.spinner("Calcul en cours..."):
            logS, desc = predict(smiles_input)

        if logS is None:
            st.error("❌ Erreur lors du calcul des descripteurs.")
        else:
            label, level   = interpret_logS(logS)
            solubility_mol = 10 ** logS

            st.divider()
            st.markdown("### 📊 Résultats")

            col1, col2 = st.columns([1, 1])

            # ── Infos moléculaires (sans image) ───────────────────────────
            with col1:
                mol     = Chem.MolFromSmiles(smiles_input)
                formula = rdMolDescriptors.CalcMolFormula(mol)
                atoms   = mol.GetNumAtoms()
                bonds   = mol.GetNumBonds()

                st.markdown("**🔬 SMILES**")
                st.code(smiles_input, language='text')
                st.markdown("**📐 Formule brute**")
                st.info(f"**{formula}**")
                st.markdown(f"⚛️ Atomes : **{atoms}** | 🔗 Liaisons : **{bonds}**")

            # ── Résultat de prédiction ─────────────────────────────────────
            with col2:
                st.metric("LogS prédit",  f"{logS:.4f}")
                st.metric("Solubilité",   f"{solubility_mol:.2e} mol/L")
                if level == "success":
                    st.success(label)
                elif level == "error":
                    st.error(label)
                else:
                    st.warning(label)

                # Échelle visuelle
                st.markdown("**Échelle de solubilité :**")
                st.markdown("""
                | LogS | Interprétation |
                |------|---------------|
                | > -1 | 🟢 Très soluble |
                | -1 à -2 | 🟡 Soluble |
                | -2 à -4 | 🟠 Peu soluble |
                | < -4 | 🔴 Très peu soluble |
                """)

            # ── Descripteurs clés ──────────────────────────────────────────
            st.divider()
            st.markdown("#### 🔬 Descripteurs moléculaires")

            key_desc = {
                "Poids mol. (g/mol)" : f"{desc['MolWt']:.2f}",
                "LogP"               : f"{desc['MolLogP']:.4f}",
                "TPSA (Ų)"          : f"{desc['TPSA']:.2f}",
                "Donneurs H"         : int(desc['NumHDonors']),
                "Accepteurs H"       : int(desc['NumHAcceptors']),
                "Liaisons rotatives" : int(desc['NumRotatableBonds']),
                "Cycles aromatiques" : int(desc['NumAromaticRings']),
                "QED"                : f"{desc['QED']:.4f}",
            }

            c1, c2 = st.columns(2)
            items  = list(key_desc.items())
            for i, (k, v) in enumerate(items):
                (c1 if i < 4 else c2).metric(k, v)

            # ── Règle de Lipinski ──────────────────────────────────────────
            st.divider()
            st.markdown("#### 💊 Règle des 5 de Lipinski")

            l1 = desc['MolWt']         <= 500
            l2 = desc['MolLogP']       <= 5
            l3 = desc['NumHDonors']    <= 5
            l4 = desc['NumHAcceptors'] <= 10

            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("MolWt ≤ 500", "✅" if l1 else "❌", f"{desc['MolWt']:.1f}")
            lc2.metric("LogP ≤ 5",    "✅" if l2 else "❌", f"{desc['MolLogP']:.2f}")
            lc3.metric("HBD ≤ 5",     "✅" if l3 else "❌", str(int(desc['NumHDonors'])))
            lc4.metric("HBA ≤ 10",    "✅" if l4 else "❌", str(int(desc['NumHAcceptors'])))

            violations = sum([not l1, not l2, not l3, not l4])
            if violations == 0:
                st.success("✅ Respecte toutes les règles de Lipinski")
            elif violations == 1:
                st.warning(f"⚠️ {violations} violation — molécule limite")
            else:
                st.error(f"❌ {violations} violations — molécule non drug-like")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()


