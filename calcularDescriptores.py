import pandas as pd
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from descriptoresRDKit import calcularDescriptoresRDKit
from descriptoresJazzy import calcularDescriptoresJazzy
from openbabel import pybel
from rdkit.Chem import CanonSmiles
from lectorParametros import LectorParametros
from sklearn.model_selection import train_test_split

pandas2ri.activate()

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)


def calcularDescriptoresCDK(smiles):
    rcdk = rpackages.importr("rcdk")
    base = rpackages.importr("base")
    descCategories = rcdk.get_desc_categories()
    getDescNames = rcdk.get_desc_names

    descNames = base.unique(base.unlist(base.sapply(descCategories,getDescNames)))
    descNames = [name for name in descNames if name != "org.openscience.cdk.qsar.descriptors.molecular.LongestAliphaticChainDescriptor"]
    descNames = ro.StrVector(descNames)

    mols = rcdk.parse_smiles(ro.StrVector(smiles))
    descriptorsCDK = rcdk.eval_desc(mols,descNames)

    descriptorsCDK = ro.conversion.rpy2py(descriptorsCDK)
    descriptorsCDK.index = [i for i in range(len(descriptorsCDK.index))]

    return descriptorsCDK




def calcularDescriptoresObabel(smiles):
    descriptorsObabel = pd.DataFrame()
    i = 0
    for smile in smiles:
        mol = pybel.readstring("smi",smile)
        desc = mol.calcdesc()
        descriptorsObabel = pd.concat([descriptorsObabel,pd.DataFrame(desc,index=[i])])
        i+=1
    descriptorsObabel.drop(columns=["cansmi","cansmiNS","formula","title","InChI","InChIKey","smarts"],inplace=True)
    return descriptorsObabel

def calcularDescriptores(smiles):
    smilesCanon = []
    i=0
    for smile in smiles:
        smilesCanon.append(CanonSmiles(smile))
    descriptorsCDK = calcularDescriptoresCDK(smilesCanon)
    descriptorsObabel = calcularDescriptoresObabel(smilesCanon)
    descriptoresRDKit = calcularDescriptoresRDKit(smilesCanon)
    descriptoresJazzy = calcularDescriptoresJazzy(smilesCanon)
    descriptors = pd.concat([descriptoresJazzy,descriptorsCDK,descriptoresRDKit,descriptorsObabel],axis=1)
    descriptors.insert(0,"smiles",smilesCanon)
    return descriptors

def crearDataset(datos,nombreArchivo,split,porcentajeSplit,prefijo=""):
    datos.sort_index(inplace=True)
    if split:
        datos_train,datos_test= train_test_split(datos,test_size=porcentajeSplit,random_state=3006)
        crearDataset(datos_train,nombreArchivo,split=False,porcentajeSplit=0,prefijo="train_")
        crearDataset(datos_test,nombreArchivo,split=False,porcentajeSplit=0,prefijo="test_")
    else:
        smiles = datos["smiles"].to_numpy()
        objetivo = datos[datos.columns[1]].to_numpy()
        dataset = calcularDescriptores(smiles)
        dataset.insert(1,datos.columns[1],objetivo)
        for index in range(2,len(datos.columns)):
            dataset.insert(2,datos.columns[index],datos[datos.columns[index]].to_numpy())
        dataset.to_csv("Datasets/"+prefijo+"desc_"+nombreArchivo,index=False)

def main():
    lector = LectorParametros()
    diccionarioValores = lector.leerParametros()
    nombreArchivoDatos = diccionarioValores["datos"]
    split = diccionarioValores["split"]
    porcentajeSplit = diccionarioValores["porcentajeSplit"]
    crearDataset(pd.read_csv("Datasets/"+nombreArchivoDatos),nombreArchivoDatos,split=split,porcentajeSplit=porcentajeSplit)

if __name__ == "__main__":
    main()