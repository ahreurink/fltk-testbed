from argparse import Namespace
from dataclasses import dataclass
from typing import List, Tuple, Type, Dict, T

import torch.distributed as dist
import torch.nn

import fltk.nets as nets
from fltk.datasets import CIFAR10Dataset, FashionMNISTDataset, CIFAR100Dataset, MNIST
from fltk.datasets.dataset import Dataset

"""
        command = (f'python3 -m fltk client {config.config_path} {task.id} '
                   f'--batchSize {cnfg.batchSize} '
                   f'--convolutionalFilters {cnfg.convolutionalFilters} '
                   f'--convolutionalLayers {cnfg.convolutionalLayers} '
                   f'--linearLayers {cnfg.linearLayers} '
                   f'--linearLayerParameters {cnfg.linearLayerParameters} '
                   f'--imageSize {cnfg.imageSize} '
                   f'--backend gloo')
                   
                   """

CLIENT_ARGS: List[Tuple[str, str, str, type]] = \
    [
     ("batchSize", "bs", "", int),
     ("convolutionalFilters", "cf", "", int),
     ("convolutionalLayers", "cl", "", int),
     ("linearLayers", "ll", "", int),
     ("linearLayerParameters", "lp", "", int),
     ("imageSize", "is", "", int),
     ]


@dataclass(frozen=True)
class LearningParameters:

    batchSize: int
    convolutionalFilters: int
    convolutionalLayers: int
    linearLayers: int
    linearLayerParameters: int
    imageSize: int

    _available_nets = {
        "CIFAR100RESNET": nets.Cifar100ResNet,
        "CIFAR100VGG": nets.Cifar100VGG,
        "CIFAR10CNN": nets.Cifar10CNN,
        "CIFAR10RESNET": nets.Cifar10ResNet,
        "FASHIONMNISTCNN": nets.FashionMNISTCNN,
        "FASHIONMNISTRESNET": nets.FashionMNISTResNet,
        'CUSTOM': nets.CustomModelMNIST
    }

    _available_data = {
        "CIFAR10": CIFAR10Dataset,
        "CIFAR100": CIFAR100Dataset,
        "FASHIONMNIST": FashionMNISTDataset,
        "MNIST": MNIST
    }

    _available_loss = {
        "CROSSENTROPY": torch.nn.CrossEntropyLoss
    }

    _available_optimizer = {
        "ADAM": torch.optim.SGD
    }

    @staticmethod
    def __safe_get(lookup: Dict[str, T], keyword: str) -> T:
        """
        Static function to 'safe' get elements from a dictionary, to prevent issues with Capitalization in the code.
        @param lookup: Lookup dictionary to 'safe get' from.
        @type lookup: dict
        @param keyword: Keyword to 'get' from the Lookup dictionary.
        @type keyword: str
        @return: Lookup value from 'safe get' request.
        @rtype: T
        """
        safe_keyword = str.upper(keyword)
        return lookup.get(safe_keyword)

    def get_model_class(self) -> Type[torch.nn.Module]:
        """
        Function to obtain the model class that was given via commandline.
        @return: Type corresponding to the model that was passed as argument.
        @rtype: Type[torch.nn.Module]
        """
        return self.__safe_get(self._available_nets, self.model)

    def get_dataset_class(self) -> Type[Dataset]:
        """
        Function to obtain the dataset class that was given via commandline.
        @return: Type corresponding to the dataset that was passed as argument.
        @rtype: Type[Dataset]
        """
        return self.__safe_get(self._available_data, self.dataset)

    def get_loss(self) -> Type:
        """
        Function to obtain the loss function Type that was given via commandline to be used during the training
        execution.
        @return: Type corresponding to the loss function that was passed as argument.
        @rtype: Type
        """
        return self.__safe_get(self._available_loss, self.loss)

    def get_optimizer(self) -> Type[torch.optim.Optimizer]:
        """
        Function to obtain the loss function Type that was given via commandline to be used during the training
        execution.
        @return: Type corresponding to the Optimizer to be used during training.
        @rtype: Type[torch.optim.Optimizer]
        """
        return self.__safe_get(self._available_optimizer, self.optimizer)


def extract_learning_parameters(args: Namespace) -> LearningParameters:
    """
    Function to extract the learning hyper-parameters from the Namespace object for the passed arguments.
    @param args: Namespace environment for running the Client.
    @type args: Namespace
    @return: Parsed learning parameters.
    @rtype: LearningParameters
    """

    batchSize = args.batchSize
    convolutionalFilters = args.convolutionalFilters
    convolutionalLayers = args.convolutionalLayers
    linearLayers = args.linearLayers
    linearLayerParameters = args.linearLayerParameters
    imageSize = args.imageSize

    return LearningParameters(batchSize, convolutionalFilters, convolutionalLayers, linearLayers, linearLayerParameters, imageSize)


def create_extractor_parser(subparsers):
    extractor_parser = subparsers.add_parser('extractor')
    extractor_parser.add_argument('config', type=str)


def create_client_parser(subparsers) -> None:
    client_parser = subparsers.add_parser('client')
    client_parser.add_argument('config', type=str)
    client_parser.add_argument('task_id', type=str)

    # Add hyper-parameters
    for long, short, hlp, tpe in CLIENT_ARGS:
        client_parser.add_argument(f'-{short}', f'--{long}', type=tpe, help=hlp, required=True)

    # Add parameter parser for backend
    client_parser.add_argument('--backend', type=str, help='Distributed backend',
                               choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                               default=dist.Backend.GLOO)


def create_cluster_parser(subparsers) -> None:
    cluster_parser = subparsers.add_parser('cluster')
    cluster_parser.add_argument('config', type=str)
    cluster_parser.add_argument('-l', '--local', type=bool, default=False)
