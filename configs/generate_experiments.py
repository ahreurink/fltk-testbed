from doepy import build
import json
import pandas as pd

DATASET = 'dataset'
CONVOLUTIONAL_LAYERS = 'convolutional_layers'
CONVOLUTIONAL_FILTERS = 'convolutional_filters'
LINEAR_LAYERS = 'linear_layers'
LINEAR_LAYERS_PARAMETERS = 'linear_layer_parameters'
CPU_CORES = 'cpu_cores'
NR_NODES = 'number_of_nodes'
BATCH_SIZE = 'batch_size'

def generate_experiment_config(experiment_table):
    experiments = [{col: val for col, val in zip(experiment_table.columns, experiment)} for experiment in experiment_table.values]
    config = [
        {
            "jobClassParameters": [
                {
                    "networkConfiguration": {
                        "network": "FashionMNISTCNN",
                        "dataset": experiment[DATASET]
                    },
                    "systemParameters": {
                        "dataParallelism": "1",
                        "executorCores": experiment[CPU_CORES],
                        "numberOfNodes": experiment[NR_NODES],
                        "executorMemory": "1Gi",
                        "action": "train"
                    },
                    "hyperParameters": {
                        "batchSize": experiment[BATCH_SIZE],
                        "convolutionalFilters": experiment[CONVOLUTIONAL_FILTERS],
                        "convolutionalLayers": experiment[CONVOLUTIONAL_LAYERS],
                        "linearLayers": experiment[LINEAR_LAYERS],
                        "linearLayerParameters": experiment[LINEAR_LAYERS_PARAMETERS],
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
