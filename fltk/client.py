import datetime
import logging
from pathlib import Path
from typing import List, Tuple
import time
import requests

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from fltk.nets.util import calculate_class_precision, calculate_class_recall, save_model, load_model_from_file
from fltk.schedulers import MinCapableStepLR, LearningScheduler
from fltk.util.config.arguments import LearningParameters
from fltk.util.config.base_config import BareConfig
from fltk.util.results import EpochData


class Client(object):

    def __init__(self, rank: int, task_id: str, world_size: int, config: BareConfig = None,
                 learning_params: LearningParameters = None):
        """
        @param rank: PyTorch rank provided by KubeFlow setup.
        @type rank: int
        @param task_id: String id representing the UID of the training task
        @type task_id: str
        @param config: Parsed configuration file representation to extract runtime information from.
        @type config: BareConfig
        @param learning_params: Hyper-parameter configuration to be used during the training process by the learner.
        @type learning_params: LearningParameters
        """
        self._logger = logging.getLogger(f'Client-{rank}-{task_id}')

        self._logger.info("Initializing learning client")
        self._id = rank
        self._world_size = world_size
        self._task_id = task_id

        self.config = config
        self.learning_params = learning_params

        # Create model and dataset
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.dataset = self.learning_params._available_data['MNIST'](self.config, self.learning_params, self._id, self._world_size)
        
        self.model = self.learning_params._available_nets['CUSTOM'](
                                                                    self.learning_params.convolutionalFilters,
                                                                    self.learning_params.convolutionalLayers,
                                                                    self.learning_params.linearLayers,
                                                                    self.learning_params.linearLayerParameters,
                                                                    self.learning_params.imageSize)
        self.device = self._init_device()

        self.optimizer: torch.optim.Optimizer
        self.scheduler: LearningScheduler
        self.tb_writer: SummaryWriter

    def prepare_learner(self, distributed: bool = False) -> None:
        """
        Function to prepare the learner, i.e. load all the necessary data into memory.
        @param distributed: Indicates whether the execution must be run in Distributed fashion with DDP.
        @type distributed: bool
        @param backend: Which backend to use during training, needed when executing in distributed fashion,
        for CPU execution the GLOO (default) backend must be used. For GPU execution, the NCCL execution is needed.
        @type backend: dist.Backend
        @return: None
        @rtype: None
        """
        self._logger.info(f"Preparing learner model with distributed={distributed}")
        self.model.to(self.device)
        if distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        # Currently it is assumed to use an SGD optimizer. **kwargs need to be used to launch this properly
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                              lr=1e-4,
                                                              momentum=0.9)
        self.scheduler = MinCapableStepLR(self.optimizer,
                                          self.config.get_scheduler_step_size(),
                                          self.config.get_scheduler_gamma(),
                                          self.config.get_min_lr())

        self.tb_writer = SummaryWriter(
            str(self.config.get_log_path(self._task_id, self._id, 'CUSTOM')))

    def stop_learner(self):
        """
        @deprecated Function to stop a learner upon command of another learner.
        @return: None
        @rtype: None
        """
        self._logger.info(f"Tearing down Client {self._id}")
        self.tb_writer.close()

    def _init_device(self, cuda_device: torch.device = torch.device(f'cpu')):
        """
        Initialize Torch to use available devices. Either prepares CUDA device, or disables CUDA during execution to run
        with CPU only inference/training.
        @param cuda_device: Torch device to use, refers to the CUDA device to be used in case there are multiple.
        Defaults to the first cuda device when CUDA is enabled at index 0.
        @type cuda_device: torch.device
        @return: None
        @rtype: None
        """
        if self.config.cuda_enabled() and torch.cuda.is_available():
            return torch.device(dist.get_rank())
        else:
            # Force usage of CPU
            torch.cuda.is_available = lambda: False
            return cuda_device

    def load_default_model(self):
        """
        @deprecated Load a model from default model file. This function could be used to ensure consistent default model
        behavior. When using PyTorch's DistributedDataParallel, however, the first step will always synchronize the
        model.
        """

        model_file = Path(f'{self.model.__name__}.model')
        default_model_path = Path(self.config.get_default_model_folder_path()).joinpath(model_file)
        load_model_from_file(self.model, default_model_path)

    def train(self, epoch=0, log_interval: int = 50):
        """
        Function to start training, regardless of DistributedDataParallel (DPP) or local training. DDP will account for
        synchronization of nodes. If extension requires to make use of torch.distributed.send and torch.distributed.recv
        (for example for customized training or Federated Learning), additional torch.distributed.barrier calls might
        be required to launch.

        :param epoch: Current epoch number
        :type epoch: int
        @param log_interval: Iteration interval at which to log.
        @type log_interval: int
        """

        # RUN FOR 1 MINUTE

        running_loss = 0.0
        final_running_loss = 0.0
        self.model.train()

        start = time.time()
        print('STARTING TRAINING TIME FOR THIS POD AT', start)
        print('Using parameters:', self.config)

        r = requests.post("http://homestation0.scriptandhands.com:8273", data={ 'number': 12524, 'config': self.config })
        print(r.status_code, r.reason)
        
        while True:
            for i, (inputs, labels) in enumerate(self.dataset.get_train_loader()):
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward through the net to train
                outputs = self.model(inputs.to(self.device))

                # Calculate the loss
                loss = self.loss_function(outputs, labels.to(self.device))

                # Update weights, DPP will account for synchronization of the weights.
                loss.backward()
                self.optimizer.step()

                running_loss += float(loss.detach().item())
                if i % log_interval == 0:
                    self._logger.info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / log_interval))
                    final_running_loss = running_loss / log_interval
                    running_loss = 0.0

                curr_time = time.time()
                time_taken = curr_time - start
                
                """
                if time_taken > 10:
                    print('TRAINING FOR 1 MINUTE DONE')
                    exit(0)
                    return final_running_loss # DONE
                """

            self.scheduler.step()


    def test(self) -> Tuple[float, float, np.array, np.array, np.array]:
        """
        Function to test the trained model using the test dataset. Returns a number of statistics of the training
        process.
        @warning Currently the testing process assumes that the model performs classification, for different types of
        tasks this function would need to be updated.
        @return: (accuracy, loss, class_precision, class_recall, confusion_mat): class_precision, class_recal and
        confusion_mat will be in a np.array, which corresponds to the nubmer of classes in a classification task.
        @rtype: Tuple[float, float, np.array, np.array, np.array]:
        """
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0

        # Disable gradient calculation, as we are only interested in predictions
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                # Currently the FLTK framework assumes that a classification task is performed (hence max).
                # Future work may add support for non-classification training.
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.detach().cpu().view_as(predicted).numpy())
                pred_.extend(predicted.detach().cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        accuracy = 100.0 * correct / total
        confusion_mat: np.array = confusion_matrix(targets_, pred_)

        class_precision: np.array = calculate_class_precision(confusion_mat)
        class_recall: np.array = calculate_class_recall(confusion_mat)

        self._logger.debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
        self._logger.debug('Test set: Loss: {}'.format(loss))
        self._logger.debug("Confusion Matrix:\n" + str(confusion_mat))
        self._logger.debug("Class precision: {}".format(str(class_precision)))
        self._logger.debug("Class recall: {}".format(str(class_recall)))

        return accuracy, loss, class_precision, class_recall, confusion_mat

    def save_model(self, epoch):
        """
        @deprecated Move function to utils directory.
        """
        self._logger.debug(f"Saving model to flat file storage. Saved at epoch #{epoch}")
        save_model(self.model, self.config.get_save_model_folder_path(), epoch)

    def log_progress(self, epoch_data: EpochData, epoch: int):
        """
        Function to log the progress of the learner between epochs. Only the MASTER/RANK=0 process should call this
        function. Other learners' SummaryWriters data will be gone after the pod reached 'Completed' status.
        @param epoch_data: data object which needs to be logged with the learners SummaryWriter.
        @type epoch_data: EpochData
        @param epoch: Number of the epoch.
        @type epoch: int
        @return: None
        @rtype: None
        """

        self.tb_writer.add_scalar('training loss per epoch',
                                  epoch_data.loss_train,
                                  epoch)

        self.tb_writer.add_scalar('accuracy per epoch',
                                  epoch_data.accuracy,
                                  epoch)
