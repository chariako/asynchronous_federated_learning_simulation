from enum import StrEnum


class ModelType(StrEnum):
    LOG_REG = "logreg"
    CNN = "cnn"
    RESNET18 = "resnet18"
    MOBILENET_V2 = "mobilenet_v2"

    @property
    def required_channels(self) -> int | None:
        """
        Returns the exact number of input channels required by the architecture.
        Returns None if the model can adapt to any input shape.
        """
        match self:
            case ModelType.RESNET18 | ModelType.MOBILENET_V2:
                return 3
            case _:
                return None

    @property
    def has_norm_layers(self) -> bool:
        """
        Returns True if the standard architecture includes normalization layers
        (BatchNorm, GroupNorm) that can be removed for stress testing.
        """
        match self:
            case ModelType.LOG_REG:
                return False
            case _:
                return True


class DeviceType(StrEnum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    AUTO = "auto"


class DatasetType(StrEnum):
    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"

    @property
    def num_classes(self) -> int:
        match self:
            case DatasetType.CIFAR100:
                return 100
            case _:
                return 10

    @property
    def num_channels(self) -> int:
        match self:
            case DatasetType.MNIST | DatasetType.FASHION_MNIST:
                return 1
            case _:
                return 3

    @property
    def image_size(self) -> int:
        """Returns the height/width (assuming square images)."""
        match self:
            case DatasetType.MNIST | DatasetType.FASHION_MNIST:
                return 28
            case _:
                return 32

    @property
    def is_grayscale(self) -> bool:
        return self.num_channels == 1


class MemoryType(StrEnum):
    DISABLED = "disabled"
    MODELS = "models"
    GRADS = "gradients"

    @property
    def requires_buffer_reset(self) -> bool:
        """
        Returns True if the strategy requires resetting
        the server's buffer after global updates.
        """
        match self:
            case MemoryType.DISABLED | MemoryType.MODELS:
                return True
            case _:
                return False

    @property
    def has_memory(self) -> bool:
        """
        Returns True if the strategy utilizes client-side memory.
        """
        match self:
            case MemoryType.GRADS | MemoryType.MODELS:
                return True
            case _:
                return False
