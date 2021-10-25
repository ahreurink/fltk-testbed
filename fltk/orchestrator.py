import logging
import time
import uuid
from queue import PriorityQueue
from typing import List
from pathlib import Path

from kubeflow.pytorchjob import PyTorchJobClient
from kubeflow.pytorchjob.constants.constants import PYTORCHJOB_GROUP, PYTORCHJOB_VERSION, PYTORCHJOB_PLURAL
from kubernetes import client

from fltk.util.cluster.client import construct_job, ClusterManager
from fltk.util.config.base_config import BareConfig
from fltk.util.task.generator.arrival_generator import ArrivalGenerator, Arrival
from fltk.util.task.task import ArrivalTask
from fltk.util.task.config.parameter import TrainTask, JobDescription, ExperimentParser, JobClassParameter


class Orchestrator(object):
    """
    Central component of the Federated Learning System: The Orchestrator

    The Orchestrator is in charge of the following tasks:
    - Running experiments
        - Creating and/or managing tasks
        - Keep track of progress (pending/started/failed/completed)
    - Keep track of timing

    Note that the Orchestrator does not function like a Federator, in the sense that it keeps a central model, performs
    aggregations and keeps track of Clients. For this, the KubeFlow PyTorch-Operator is used to deploy a train task as
    a V1PyTorchJob, which automatically generates the required setup in the cluster. In addition, this allows more Jobs
    to be scheduled, than that there are resources, as such, letting the Kubernetes Scheduler let decide when to run
    which containers where.
    """
    _alive = False
    # Priority queue, requires an orderable object, otherwise a Tuple[int, Any] can be used to insert.
    pending_tasks: "PriorityQueue[ArrivalTask]" = PriorityQueue()
    deployed_tasks: List[ArrivalTask] = []
    completed_tasks: List[str] = []

    def __init__(self, cluster_mgr: ClusterManager, arv_gen: ArrivalGenerator, config: BareConfig):
        self.__logger = logging.getLogger('Orchestrator')
        self.__logger.debug("Loading in-cluster configuration")
        self.__cluster_mgr = cluster_mgr
        self.__arrival_generator = arv_gen
        self._config = config

        # API to interact with the cluster.
        self.__client = PyTorchJobClient()

    def stop(self) -> None:
        """
        Stop the Orchestrator.
        @return:
        @rtype:
        """
        self.__logger.info("Received stop signal for the Orchestrator.")
        self._alive = False

    def run(self, clear: bool = True) -> None:
        """
        Main loop of the Orchestartor.
        @param clear: Boolean indicating whether a previous deployment needs to be cleaned up (i.e. lingering jobs that
        were deployed by the previous run).

        @type clear: bool
        @return: None
        @rtype: None
        """
        print('Starting loop...')
        self._alive = True

        CPU_CORES = 2
        NUMBER_OF_NODES = 4

        pending = []
        config: Path = Path('configs/tasks/arrival_config.json')

        print('Loading train jobs...')
        # load all jobs that we are going to run in this session
        parser = ExperimentParser(config_path=config)
        experiment_descriptions = parser.parse()
        self.job_dict = {f'train_job_{indx}': item for indx, item in enumerate(experiment_descriptions)}
        print('Loading job dict with size', len(self.job_dict))
        print()

        for x in self.job_dict:
            print(x)
            print()

        print('Start processing!')
        for job in self.job_dict.values():

            for parameters in job.job_class_parameters:
                self.__clear_jobs()
                print('PARAMS', parameters)

                priority = parameters.priorities[0]

                inter_arrival_ticks = 0
                task_id = 0
                train_task = TrainTask(task_id, parameters, priority)

                arrival = Arrival(inter_arrival_ticks, train_task, task_id)

                unique_identifier: uuid.UUID = uuid.uuid4()
                task = ArrivalTask(priority=arrival.get_priority(),
                                    id=unique_identifier,
                                    network=arrival.get_network(),
                                    dataset=arrival.get_dataset(),
                                    sys_conf=arrival.get_system_config(),
                                    param_conf=arrival.get_parameter_config())
                print(f"Arrival of: {task}")
                
                job_to_start = construct_job(self._config, task)


                # Hack to overcome limitation of KubeFlow version (Made for older version of Kubernetes)
                self.__logger.info(f"Deploying on cluster: {task.id}")
                self.__client.create(job_to_start, namespace=self._config.cluster_config.namespace)
                self.__logger.info("Creation done, shutting down...")

                self.stop()
                return

            self.__logger.debug("Still alive...")
            time.sleep(5)

        logging.info(f'Experiment completed, currently does not support waiting.')

    def __clear_jobs(self):
        """
        Function to clear existing jobs in the environment (i.e. old experiments/tests)
        @return: None
        @rtype: None
        """
        namespace = self._config.cluster_config.namespace
        self.__logger.info(f'Clearing old jobs in current namespace: {namespace}')

        for job in self.__client.get(namespace=self._config.cluster_config.namespace)['items']:
            job_name = job['metadata']['name']
            self.__logger.info(f'Deleting: {job_name}')
            try:
                self.__client.custom_api.delete_namespaced_custom_object(
                    PYTORCHJOB_GROUP,
                    PYTORCHJOB_VERSION,
                    namespace,
                    PYTORCHJOB_PLURAL,
                    job_name)
            except Exception as e:
                self.__logger.warning(f'Could not delete: {job_name}')
                print(e)
