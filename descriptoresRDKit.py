from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.EState import EState_VSA
import pandas as pd
import numpy


def calcularDescriptores(molecula):
    descriptores = {}

    descriptores["ExactMolWeight"] = Descriptors.ExactMolWt(molecula)
    descriptores["FpDensityMorgan1"] = Descriptors.FpDensityMorgan1(molecula)
    descriptores["FpDensityMorgan2"] = Descriptors.FpDensityMorgan2(molecula)
    descriptores["FpDensityMorgan3"] = Descriptors.FpDensityMorgan3(molecula)

    descriptores["HeavyAtomMolWt"] = Descriptors.HeavyAtomMolWt(molecula)

    descriptores["MaxAbsPartialCharge"] = Descriptors.MaxAbsPartialCharge(molecula)
    descriptores["MaxPartialCharge"] = Descriptors.MaxPartialCharge(molecula)
    descriptores["MinAbsPartialCharge"] = Descriptors.MinAbsPartialCharge(molecula)
    descriptores["MinPartialCharge"] = Descriptors.MinPartialCharge(molecula)

    descriptores["MolWt"] = Descriptors.MolWt(molecula)
    descriptores["NumRadicalElectrons"] = Descriptors.NumRadicalElectrons(molecula)
    descriptores["NumValenceElectrons"] = Descriptors.NumValenceElectrons(molecula)

    descriptores["BalabanJ"] = GraphDescriptors.BalabanJ(molecula)
    descriptores["BertzCT"] = GraphDescriptors.BertzCT(molecula)

    descriptores["Chi0"] = GraphDescriptors.Chi0(molecula)
    descriptores["Chi1"] = GraphDescriptors.Chi1(molecula)

    descriptores["Chi0n"] = GraphDescriptors.Chi0n(molecula)
    descriptores["Chi1n"] = GraphDescriptors.Chi1n(molecula)
    descriptores["Chi2n"] = GraphDescriptors.Chi2n(molecula)
    descriptores["Chi3n"] = GraphDescriptors.Chi3n(molecula)
    descriptores["Chi4n"] = GraphDescriptors.Chi4n(molecula)

    descriptores["Chi0v"] = GraphDescriptors.Chi0v(molecula)
    descriptores["Chi1v"] = GraphDescriptors.Chi1v(molecula)
    descriptores["Chi2v"] = GraphDescriptors.Chi2v(molecula)
    descriptores["Chi3v"] = GraphDescriptors.Chi3v(molecula)
    descriptores["Chi4v"] = GraphDescriptors.Chi4v(molecula)

    descriptores["HallKierAlpha"] = GraphDescriptors.HallKierAlpha(molecula)

    descriptores["Kappa1"] = GraphDescriptors.Kappa1(molecula)
    descriptores["Kappa2"] = GraphDescriptors.Kappa2(molecula)
    descriptores["Kappa3"] = GraphDescriptors.Kappa3(molecula)

    descriptores["MolLogP"] = Crippen.MolLogP(molecula)
    descriptores["MolMR"] = Crippen.MolMR(molecula)

    descriptores["HeavyAtomCount"] = Lipinski.HeavyAtomCount(molecula)
    descriptores["NHOHCount"] = Lipinski.NHOHCount(molecula)
    descriptores["NOCount"] = Lipinski.NOCount(molecula)
    descriptores["NumHAcceptors"] = Lipinski.NumHAcceptors(molecula)
    descriptores["NumHDonors"] = Lipinski.NumHDonors(molecula)
    descriptores["NumHeteroatoms"] = Lipinski.NumHeteroatoms(molecula)
    descriptores["NumRotatableBonds"] = Lipinski.NumRotatableBonds(molecula)
    descriptores["NumAmideBonds"] = rdMolDescriptors.CalcNumAmideBonds(molecula)

    descriptores["NumAromaticRings"] = Lipinski.NumAromaticRings(molecula)
    descriptores["NumAliphaticRings"] = Lipinski.NumAliphaticRings(molecula)
    descriptores["NumSaturatedRings"] = Lipinski.NumSaturatedRings(molecula)

    descriptores["NumAliphaticCarbocycles"] = Lipinski.NumAliphaticCarbocycles(molecula)
    descriptores["NumAliphaticHeterocycles"] = Lipinski.NumAliphaticHeterocycles(molecula)

    descriptores["NumAromaticCarbocycles"] = Lipinski.NumAromaticCarbocycles(molecula)
    descriptores["NumAromaticHeterocycles"] = Lipinski.NumAromaticHeterocycles(molecula)

    descriptores["NumSaturatedCarbocycles"] = Lipinski.NumSaturatedCarbocycles(molecula)
    descriptores["NumSaturatedHeterocycles"] = Lipinski.NumSaturatedHeterocycles(molecula)

    descriptores["RingCount"] = Lipinski.RingCount(molecula)
    descriptores["FractionCSP3"] = Lipinski.FractionCSP3(molecula)

    descriptores["NumSpiroAtoms"] = rdMolDescriptors.CalcNumSpiroAtoms(molecula)
    descriptores["NumBridgeheadAtoms"] = rdMolDescriptors.CalcNumBridgeheadAtoms(molecula)
    descriptores["TPSA"] = rdMolDescriptors.CalcTPSA(molecula)
    descriptores["LabuteASA"] = rdMolDescriptors.CalcLabuteASA(molecula)
    
    PEOE_VSA = rdMolDescriptors.PEOE_VSA_(molecula)
    for i in range(1,15):
        descriptores["PEOE_VSA_"+str(i)] = PEOE_VSA[i-1]

    SMR_VSA = rdMolDescriptors.SMR_VSA_(molecula)
    for i in range(1,11):
        descriptores["SMR_VSA_"+str(i)] = SMR_VSA[i-1]

    SlogP_VSA = rdMolDescriptors.SlogP_VSA_(molecula)
    for i in range(1,11):
        descriptores["SlogP_VSA_"+str(i)] = SlogP_VSA[i-1]
    
    
    MQNs = rdMolDescriptors.MQNs_(molecula)
    for i in range(1,43):
        descriptores["MQNs_"+str(i)] = MQNs[i-1]

    return descriptores

def calcularDescriptores2(molecula):
    return Chem.Descriptors.CalcMolDescriptors(molecula)


def calcularDescriptoresRDKit(smiles):
    descriptoresRDKit = {}
    for smile in smiles:
        try:
            molecula = Chem.MolFromSmiles(smile)
            descriptores = calcularDescriptores(molecula)
            for descriptor in descriptores:
                if descriptor in descriptoresRDKit:
                    descriptoresRDKit[descriptor].append(descriptores[descriptor])
                else:
                    descriptoresRDKit[descriptor] = [descriptores[descriptor]]
        except:
            for descriptor in descriptoresRDKit:
                descriptoresRDKit[descriptor].append("")
    return pd.DataFrame(descriptoresRDKit)


