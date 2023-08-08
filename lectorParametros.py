import sys
import getopt

class LectorParametros:

    def __init__(self):
        self.argumentList = sys.argv[2:]
        self.options = "d:c:t:s:"
        self.longOptions = ["dataset =","cross-validation =","tipo =","split ="]
    
    def leerParametros(self):
        parametros, valores = getopt.getopt(self.argumentList, self.options, self.longOptions)
        diccionarioValores = {}
        splitFlag = False
        porcentajeSplit = 0.15
        for parametroActual, valorActual in  parametros:
            if parametroActual in ("-d","--dataset"):
                diccionarioValores["datos"] = valorActual
            elif parametroActual in ("-c","--cross-validation"):
                diccionarioValores["validacion"] = valorActual
            elif parametroActual in ("-t","--tipo"):
                diccionarioValores["tipo"] = valorActual
            elif parametroActual in ("-s","--split"):
                splitFlag = True
                porcentajeSplit = float(valorActual)
        diccionarioValores["split"] = splitFlag
        diccionarioValores["porcentajeSplit"] = porcentajeSplit
        return diccionarioValores

