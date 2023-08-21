import sys
from calcularDescriptores import main as crearDatasetMain
from procesarDatos import main as procesarDatosMain
from busqueda import main as busquedaMain
from guardarModelos import main as guardarModelosMain
from evaluar import main as evaluarMain

        
if __name__=="__main__":
    modo = sys.argv[1]
    if modo == "desc":
        crearDatasetMain()
    elif modo == "proc":
        procesarDatosMain()
    elif modo == "search":
        busquedaMain()
    elif modo == "train":
        guardarModelosMain()
    elif modo == "evaluar":
        evaluarMain()

