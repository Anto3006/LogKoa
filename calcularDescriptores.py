import pandas
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from descriptoresRDKit import calcularDescriptoresRDKit
from descriptoresJazzy import calcularDescriptoresJazzy
from openbabel import pybel
import warnings

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

    mols = rcdk.parse_smiles(ro.StrVector(smiles))
    descriptorsCDK = rcdk.eval_desc(mols,descNames)

    descriptorsCDK = ro.conversion.rpy2py(descriptorsCDK)
    descriptorsCDK.index = [i for i in range(len(descriptorsCDK.index))]

    return descriptorsCDK




def calcularDescriptoresObabel(smiles):
    descriptorsObabel = pandas.DataFrame()
    i = 0
    for smile in smiles:
        mol = pybel.readstring("smi",smile)
        desc = mol.calcdesc()
        descriptorsObabel = pandas.concat([descriptorsObabel,pandas.DataFrame(desc,index=[i])])
        i+=1
    descriptorsObabel.drop(columns=["cansmi","cansmiNS","formula","title","InChI","InChIKey","smarts","s","L5"],inplace=True)
    return descriptorsObabel

def calcularDescriptores(smiles):
    descriptorsCDK = calcularDescriptoresCDK(smiles)
    descriptorsObabel = calcularDescriptoresObabel(smiles)
    descriptoresRDKit = calcularDescriptoresRDKit(smiles)
    descriptoresJazzy = calcularDescriptoresJazzy(smiles)
    descriptors = pandas.concat([descriptoresJazzy,descriptorsCDK,descriptoresRDKit,descriptorsObabel],axis=1)
    descriptors.insert(0,"smiles",smiles)
    return descriptors


