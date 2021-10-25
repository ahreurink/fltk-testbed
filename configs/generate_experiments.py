from doepy import build
import json
import pandas as pd

CONVOLUTIONAL_LAYERS = 'convolutionalLayers'
CONVOLUTIONAL_FILTERS = 'convolutionalFilters'
LINEAR_LAYERS = 'linearLayers'
LINEAR_LAYERS_PARAMETERS = 'linearLayerParameters'
CORES = 'coresPerNode'
NR_NODES = 'dataParallelism'
BATCH_SIZE = 'batchSize'
IMAGE_SIZE = 'imageSize'

def generate_experiment_config(experiment_table):
    experiments = [{col: val for col, val in zip(experiment_table.columns, experiment)} for experiment in experiment_table.values]
    print(experiments)
    config = [
        {
            "jobClassParameters": [
                {
                    "networkConfiguration": {
                        "network": "NONE",
                        "dataset": "NONE"
                    },
                    "systemParameters": {
                        "dataParallelism": int(experiment[NR_NODES]),
                        "executorCores": "750m",
                        "executorMemory": "1Gi",
                        "action": "train",
                        "coresPerNode": int(experiment[CORES])
                    },
                    "hyperParameters": {
                        "batchSize": int(experiment[BATCH_SIZE]),
                        "convolutionalFilters": int(experiment[CONVOLUTIONAL_FILTERS]),
                        "convolutionalLayers": int(experiment[CONVOLUTIONAL_LAYERS]),
                        "linearLayers": int(experiment[LINEAR_LAYERS]),
                        "linearLayerParameters": int(experiment[LINEAR_LAYERS_PARAMETERS]),
                        "imageSize": int(experiment[IMAGE_SIZE]),
                        "maxEpoch": "5",
                        "learningRate": "0.01",
                        "learningrateDecay": "0.0002"
                    },
                    "classProbability": 0.1,
                    "priorities": [
                        {
                            "priority": 1,
                            "probability": 1
                        }
                    ]
                }
            for experiment in experiments],
            "lambda": 20,
            "preemptJobs": 0
        }
    ]
    f = open("config.json", "w")
    print('DUMPING', config)
    f.write(json.dumps(config))
    f.close()


def generate_table(parameters):
    table = build.frac_fact_res(parameters, 5)
    for key in parameters:
        if not parameters[key][0].isnumeric():
            table[key].replace(0, parameters[key][0], True)
            table[key].replace(1, parameters[key][1], True)
    return table


def main():
    f = open('parameters.txt')
    lines = f.readlines()
    parameters = {key: [value_min, value_max] for key, value_min, value_max in
                  [line.strip().split(',') for line in lines]}
    signs = {key: ['-', '+'] for key, _, _ in [line.strip().split(',') for line in lines]}

    experiments = generate_table(parameters)
    sign_table = generate_table(signs)
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(sign_table)
    generate_experiment_config(experiments)


if __name__ == '__main__':
    main()
