from enum import StrEnum


class ModelType(StrEnum):
    """Enumeration of supported neural network architectures."""

    LOG_REG = "logreg"
    CNN = "cnn"
    RESNET18 = "resnet18"

    @property
    def required_channels(self) -> int | None:
        """
        Determines the strict number of input channels required by the architecture.

        Returns:
            int | None: The required integer channel count, or None if the model
                dynamically adapts to any input shape.
        """
        match self:
            case ModelType.RESNET18:
                return 3
            case _:
                return None


class DeviceType(StrEnum):
    """Enumeration of supported hardware accelerator backends."""

    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    AUTO = "auto"


class DatasetType(StrEnum):
    """Enumeration of supported federated learning datasets."""

    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"

    @property
    def train_size(self) -> int:
        """
        Retrieves the total number of samples in the raw training split.

        Returns:
            int: The training dataset size.
        """
        match self:
            case DatasetType.MNIST | DatasetType.FASHION_MNIST:
                return 60000
            case DatasetType.CIFAR10 | DatasetType.CIFAR100:
                return 50000

    @property
    def test_size(self) -> int:
        """
        Retrieves the total number of samples in the raw evaluation split.

        Returns:
            int: The evaluation dataset size.
        """
        return 10000

    @property
    def num_classes(self) -> int:
        """
        Retrieves the total number of target classes/labels in the dataset.

        Returns:
            int: The number of classes.
        """
        match self:
            case DatasetType.CIFAR100:
                return 100
            case _:
                return 10

    @property
    def num_channels(self) -> int:
        """
        Retrieves the number of color channels in the dataset images.

        Returns:
            int: 1 for grayscale, 3 for RGB.
        """
        match self:
            case DatasetType.MNIST | DatasetType.FASHION_MNIST:
                return 1
            case _:
                return 3

    @property
    def image_size(self) -> int:
        """
        Retrieves the pixel height and width of the images (assumes square aspect ratio).

        Returns:
            int: The image dimension in pixels.
        """
        match self:
            case DatasetType.MNIST | DatasetType.FASHION_MNIST:
                return 28
            case _:
                return 32

    @property
    def is_grayscale(self) -> bool:
        """
        Determines if the dataset images are grayscale.

        Returns:
            bool: True if the dataset has 1 channel, False otherwise.
        """
        return self.num_channels == 1


class MemoryType(StrEnum):
    """Enumeration of client-side memory tracking strategies."""

    DISABLED = "disabled"
    MODELS = "models"
    GRADS = "gradients"

    @property
    def requires_buffer_reset(self) -> bool:
        """
        Determines if the server buffer must be flushed after a global update.

        Returns:
            bool: True if the server buffer requires resetting, False if updates
                are accumulated continuously (e.g., gradient memory).
        """
        match self:
            case MemoryType.GRADS:
                return False
            case _:
                return True

    @property
    def has_memory(self) -> bool:
        """
        Determines if the selected strategy necessitates tracking client-side memory states.

        Returns:
            bool: True if memory is actively used, False if disabled.
        """
        match self:
            case MemoryType.DISABLED:
                return False
            case _:
                return True
