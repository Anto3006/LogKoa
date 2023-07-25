import pandas
import cirpy
from rdkit import Chem
from rdkit.Chem import Crippen
from jazzy.api import deltag_from_smiles
from jazzy.api import molecular_vector_from_smiles
import warnings

nombresColumnas = {"sdc":"CHds","sdx":"XHds","sa":"HBAs"}
nombresColumnasEnergias = {"dga":"HydA","dgp":"HydP","dgtot":"Hyd"}

def filterDataByTemp(data,temp):
    filtered_data = data[data["Temp"]==temp]
    filtered_data.reset_index()
    return filtered_data

def addSmilesToDataByCas(data):
    smiles = []
    for cas in data["Cas_No"]:
        smiles.append(cirpy.resolve(cas,"smiles"))
    data["smiles2"] = smiles
    return data

def calcularDescriptoresJazzy(smiles):
    warnings.filterwarnings("ignore")
    deltag_sol_oct0 = []
    deltag_sol_oct = []
    v_dict = {}
    descriptores = {}
    for smile in smiles:
        try:
            smile2 = Chem.CanonSmiles(smile)
            m = Chem.MolFromSmiles(smile2)
            v = molecular_vector_from_smiles(smile2)
            for key in nombresColumnas:
                if nombresColumnas[key] in v_dict:
                    v_dict[nombresColumnas[key]].append(v[key])
                else:
                    v_dict[nombresColumnas[key]] = [v[key]]
            for key in nombresColumnasEnergias:
                if nombresColumnasEnergias[key] in v_dict:
                    v_dict[nombresColumnasEnergias[key]+"0"].append(v[key])
                    v_dict[nombresColumnasEnergias[key]].append(v[key]/4.184)
                else:
                    v_dict[nombresColumnasEnergias[key]+"0"] = [v[key]]
                    v_dict[nombresColumnasEnergias[key]] = [v[key]/4.184]
            logP = Chem.Crippen.MolLogP(m)
            d0 = deltag_from_smiles(smile2)
            d = d0/4.184 
            dg_sol_oct0 = -logP*1.36+d0
            dg_sol_oct = -logP*1.36+d
            deltag_sol_oct.append(dg_sol_oct)
        except Exception as e:
            print(e)
            deltag_sol_oct.append(" ")
            for key in nombresColumnas:
                if nombresColumnas[key] in v_dict:
                    v_dict[nombresColumnas[key]].append(" ")
                else:
                    v_dict[nombresColumnas[key]] = [" "]
            for key in nombresColumnasEnergias:
                if nombresColumnasEnergias[key] in v_dict:
                    v_dict[nombresColumnasEnergias[key]+"0"].append(" ")
                    v_dict[nombresColumnasEnergias[key]].append(" ")
                else:
                    v_dict[nombresColumnasEnergias[key]+"0"] = [" "]
                    v_dict[nombresColumnasEnergias[key]] = [" "]
            print("Error:",smile)
    for key in v_dict:
        descriptores[key] = v_dict[key]
    descriptores["delta_g_sol_oct"] = deltag_sol_oct
    return pandas.DataFrame(descriptores)


def crearBaseConSmiles(nombreArchivoBaseOriginal):
    """
    No se calculan todos los SMILES, hay que agregar algunos manualmente
    """
    datos = pandas.read_csv(archivoBaseOriginal)
    datos_25C = filterDataByTemp(datos,25)
    datos_25C_smiles = addSmilesToDataByCas(datos_25C)
    columnas = ["ChemID","Cas_No","Chemical_Name","Temp","log_KOA","smiles"]
    columnas_datos = datos_25C_smiles.columns.values
    for col in columnas_datos:
        if not col in columnas:
            datos_25C_smiles.drop(columns=col,inplace=True)
    datos_25C_smiles.to_csv("logKoa_SMILES.csv")

