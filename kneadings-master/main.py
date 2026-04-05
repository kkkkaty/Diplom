import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib/computation_template'))

import src.computing.workers_kneadings_pendulums as wrk
import src.computing.engines_kneadings_fbpo as engine
from lib.computation_template.engine import workflow, getConfiguration, parseArguments

ENGINE_REGISTRY = {'kneadings_pendulums': engine.general_engine}

if __name__ == "__main__":
    parseArguments(sys.argv)
    configDict = getConfiguration(sys.argv[1])
    taskName = configDict['task']
    initFunc = wrk.registry['init'][taskName]
    worker = wrk.registry['worker'][taskName]
    engine = ENGINE_REGISTRY[taskName]
    postProcess = wrk.registry['post'][taskName]
    def gridMaker(configDict): pass
    workflow(configDict, initFunc, gridMaker, worker, engine, postProcess)

