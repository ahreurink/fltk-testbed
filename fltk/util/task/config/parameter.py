import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dataclasses_json import config, dataclass_json


@dataclass_json
@dataclass(frozen=True)
class HyperParameters:
    """
    Learning HyperParameters.
        "batchSize": experiment[BATCH_SIZE],
        "convolutionalFilters": experiment[CONVOLUTIONAL_FILTERS],
        "convolutionalLayers": experiment[CONVOLUTIONAL_LAYERS],
        "linearLayers": experiment[LINEAR_LAYERS],
        "linearLayerParameters": experiment[LINEAR_LAYERS_PARAMETERS],
        "imageSize": experiment[IMAGE_SIZE],
    """
    batchSize: int = field(metadata=config(field_name="batchSize"))
    convolutionalFilters: int = field(metadata=config(field_name="convolutionalFilters"))
    convolutionalLayers: int = field(metadata=config(field_name="convolutionalLayers"))
    linearLayers: int = field(metadata=config(field_name="linearLayers"))
    linearLayerParameters: int = field(metadata=config(field_name="linearLayerParameters"))
    imageSize: int = field(metadata=config(field_name="imageSize"))

@dataclass_json
@dataclass(frozen=True)
class Priority:
    """
    Job class priority, indicating the presedence of one arrival over another.
    """
    priority: int
    probability: float


@dataclass_json
@dataclass(frozen=True)
class SystemParameters:
    """
    System parameters to spawn pods with.
    data_parallelism: Number of pods (distributed) that will work together on training the network.
    executor_cores: Number of cores assigned to each executor.
    executor_memory: Amount of RAM allocated to each executor.
    action: Indicating whether it regards 'inference' or 'train'ing time.
    """
    data_parallelism: int = field(metadata=config(field_name="dataParallelism"))
    executor_cores: int = field(metadata=config(field_name="executorCores"))
    executor_memory: str = field(metadata=config(field_name="executorMemory"))
    action: str = field(metadata=config(field_name="action"))
    cores_per_node: str = field(metadata=config(field_name="coresPerNode"))


@dataclass_json
@dataclass(frozen=True)
class NetworkConfiguration:
    """
    Dataclass describing the network and dataset that is 'trained' for a task.
    """
    network: str
    dataset: str


@dataclass_json
@dataclass(frozen=True)
class JobClassParameter:
    """
    Dataclass describing the job specific parameters (system and hyper).
    """
    network_configuration: NetworkConfiguration = field(metadata=config(field_name="networkConfiguration"))
    system_parameters: SystemParameters = field(metadata=config(field_name="systemParameters"))
    hyper_parameters: HyperParameters = field(metadata=config(field_name="hyperParameters"))
    class_probability: float = field(metadata=config(field_name="classProbability"))
    priorities: List[Priority] = field(metadata=config(field_name="priorities"))


@dataclass_json
@dataclass(frozen=True)
class JobDescription:
    """
    Dataclass describing the characteristics of a Job type, as well as the corresponding arrival statistic.
    Currently, the arrival statistics is the lambda value used in a Poisson arrival process.

    preemtible_jobs: indicates whether the jobs can be pre-emptively rescheduled by the scheduler. This is currently
    not implemented in FLTK, but could be added as a project (advanced)
    """
    job_class_parameters: List[JobClassParameter] = field(metadata=config(field_name="jobClassParameters"))
    arrival_statistic: float = field(metadata=config(field_name="lambda"))
    preemtible_jobs: float = field(metadata=config(field_name="preemptJobs"))


@dataclass(order=True)
class TrainTask:
    """
    Training description used by the orchestrator to generate tasks. Contains 'transposed' information of the
    configuration file to make job generation easier and cleaner by using a 'flat' data class.

    Dataclass is ordered, to allow for ordering of arrived tasks in a PriorityQueue (for scheduling).
    """
    priority: int
    network_configuration: NetworkConfiguration = field(compare=False)
    system_parameters: SystemParameters = field(compare=False)
    hyper_parameters: HyperParameters = field(compare=False)
    identifier: str = field(compare=False)

    def __init__(self, identity: str, job_parameters: JobClassParameter, priority: Priority):
        """
        Overridden init method for dataclass, to allow for 'exploding' a JobDescription object to a flattened object.
        @param job_parameters:
        @type job_parameters:
        @param job_description:
        @type job_description:
        @param priority:
        @type priority:
        """
        self.identifier = identity
        self.network_configuration = job_parameters.network_configuration
        self.system_parameters = job_parameters.system_parameters
        self.hyper_parameters = job_parameters.hyper_parameters
        self.priority = priority.priority


class ExperimentParser(object):

    def __init__(self, config_path: Path):
        self.__config_path = config_path

    def parse(self) -> List[JobDescription]:
        """
        Parse function to load JSON conf into JobDescription objects. Any changes to the JSON file format
        should be reflected by the classes used. For more information refer to the dataclasses JSON
        documentation https://pypi.org/project/dataclasses-json/.
        """
        print('parsing...')
        with open(self.__config_path, 'r') as config_file:
            config_dict = json.load(config_file)
            print('DICT', config_dict)
            def x(job_description):
                print('LOADING DICT DATA', job_description)
                return JobDescription.from_dict(job_description)
            job_list = [x(job_description) for job_description in config_dict]
        return job_list
