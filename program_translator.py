
class ProgramTranslator(object):
    def __init__(self, programDict, maxArity):
        self.programDict = programDict
        self.maxArity = maxArity

        self.maxStack = 0

    def functionToKey(self, function, withValInputs = True):
        valInputs = ""
        if withValInputs:
            valInputs = "_" + ",".join(function["value_inputs"])
        functionKey = function["function"] if "_" in function["function"] else \
                      "_".join([function["function"], function["function"]])
        return str(len(function["inputs"])) + "_" + functionKey + valInputs

    def keyToFunction(self, key):
        assert key not in self.programDict.invalidSymbols
        function = {}
        parts = key.split("_")
        arity = int(parts[0])
        function["function"] = "_".join([parts[1], parts[2]])
        function["value_inputs"] = []
        if len(parts) == 4:
            function["value_inputs"] = parts[3].split(",")
        function["inputs"] = []
        return function, arity
    
    def keyToArity(self, key):
        if key in self.programDict.invalidSymbols:
            return 0
        return int(key.split("_")[0])

    def keyToType(self, key):
        if key in self.programDict.invalidSymbols:
            return ["0", "0", "0"]
        return ["0:" + key.split("_")[0], "1:" + key.split("_")[1], "2:" + key.split("_")[2]]

    def programToPostfixProgram(self, program):
        newProgram = []
        
        def programToPostfixAux(currIndex = -1):
            childrenIndices = program[currIndex]["inputs"]
            #[int(child) for child in program[currIndex]["inputs"]]
            childrenNewIndices = []
            for child in childrenIndices:
                programToPostfixAux(child)
                childrenNewIndices.append(len(newProgram) - 1)
            program[currIndex]["inputs"] = childrenNewIndices
            newProgram.append(program[currIndex])
        
        programToPostfixAux()
        return newProgram

    def programToSeq(self, program):
        return [self.functionToKey(function) for function in program]

    def programToInputs(self, program, offset = 0):
        inputs = [function["inputs"] for function in program]
        offsetedInputs = [[FuncInput + offset for FuncInput in FuncInputs] for FuncInputs in inputs]
        return offsetedInputs

    # def seqToProgram(self, seq, enforceValidPrograms = True):
    #     program = []

    #     def seqToProgramAux(currIndex = len(seq) - 1):
    #         if currIndex < 0:
    #             program = None
    #             return
    #         currFunc, arity = self.keyToFunction(seq[currIndex])
    #         nextIndex = currIndex - 1
    #         program.append(currFunc)
    #         for _ in arity:
    #             currFunc["inputs"].append(nextIndex)
    #             nextIndex = seqToProgramAux(nextIndex)
    #         currFunc["inputs"].reverse()
    #         return nextIndex

    #     if enforceValidPrograms:
    #         seqToProgramAux()
    #         if program is not None:
    #             program.reverse()
    #     else:
    #         stack = [0] * self.maxArity
    #         for i in range(len(seq)):
    #             func, arity = self.keyToFunction(seq[i])
    #             func["inputs"] = stack[len(stack) - arity:]
    #             newLength = max(len(stack) - arity, self.maxArity)
    #             stack = stack[:newLength] + [i + self.maxArity]
    #             self.maxStack = max(len(stack), self.maxStack)
    #             program.append(func)

    #     return program
