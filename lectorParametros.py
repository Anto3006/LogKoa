import sys
import getopt

class LectorParametros:

    def __init__(self):
        self.argumentList = sys.argv[1:]
        self.options = "d:c:ct:t:s:"
        self.longOptions = ["dataset =","cross-validation =","tipo =","split =", "const =", "corr =", "cvThreshold ="]
    
    def leerParametros(self):
        parametros, valores = getopt.getopt(self.argumentList, self.options, self.longOptions)
        diccionarioValores = {}
        splitFlag = False
        porcentajeSplit = 0.15
        constThreshold = 0.8
        corrThreshold = 0.95
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
            elif parametroActual == "--const":
                diccionarioValores["constThreshold"] = valorActual
            elif parametroActual == "--corr":
                diccionarioValores["corrThreshold"] = valorActual
            elif parametroActual in ("-ct","--cvThreshold"):
                diccionarioValores["cvThreshold"] = valorActual
        diccionarioValores["split"] = splitFlag
        diccionarioValores["porcentajeSplit"] = porcentajeSplit
        diccionarioValores["constThreshold"] = constThreshold
        diccionarioValores["corrThreshold"] = corrThreshold
        return diccionarioValores

