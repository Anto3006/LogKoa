import sys
import getopt
import json
import re

class LectorParametros:

    def __init__(self):
        self.argumentList = sys.argv[1:]
        self.options = "d:c:t:s:h:"
        self.longOptions = ["dataset=","cross-validation=","tipo=","split=", "colNA=","const=", "corr=", "cvThreshold="]
        opcionesLector = json.load(open("parametrosLineaComando.json","r"))
        self.parameterMap = opcionesLector["llaves"]
        self.defaultValues = opcionesLector["valores_default"]

    def leerParametros(self):
        parametros, valores = getopt.getopt(self.argumentList, self.options, self.longOptions)
        diccionarioValores = {}
        for parametroActual, valorActual in  parametros:
            llaveActual = self.parameterMap[parametroActual]
            if llaveActual not in diccionarioValores:
                isNumber = re.search("^[0-9]+.?[0-9]*$",valorActual)
                if isNumber:
                    if not "." in valorActual:
                        valorActual = int(valorActual)
                    else:
                        valorActual = float(valorActual)
                diccionarioValores[llaveActual] = valorActual
                if llaveActual == "porcentajeSplit":
                    diccionarioValores["split"] = True
            else:
                raise Exception(f"Este argumento ya fue utilizado: {parametroActual}")
        for parametro in self.parameterMap:
            llave = self.parameterMap[parametro]
            if llave not in diccionarioValores:
                if llave == "porcentajeSplit":
                    diccionarioValores["split"] = True
                if llave in self.defaultValues:
                    diccionarioValores[llave] = self.defaultValues[llave]
                else:
                    diccionarioValores[llave] = None

        return diccionarioValores

