import sys
import getopt

class LectorParametros:

    def __init__(self):
        self.argumentList = sys.argv[2:]
        self.options = "d:c:t:"
        self.longOptions = ["dataset =","cross-validation =","tipo ="]
    
    def leerParametros(self):
        parametros, valores = getopt.getopt(self.argumentList, self.options, self.longOptions)
        diccionarioValores = {}
        for parametroActual, valorActual in  parametros:
            if parametroActual in ("-d","--dataset"):
                diccionarioValores["datos"] = valorActual
            elif parametroActual in ("-c","--cross-validation"):
                diccionarioValores["validacion"] = valorActual
            if parametroActual in ("-t","--tipo"):
                diccionarioValores["tipo"] = valorActual
        return diccionarioValores

