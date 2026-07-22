from enum import StrEnum


class ModelType(StrEnum):
    """Enumeration of supported neural network architectures.

    Attributes:
        LOG_REG: Logistic regression architecture.
        CNN: Simple Convolutional Neural Network architecture.
        RESNET18: ResNet-18 architecture.
    """

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
    """Enumeration of supported hardware accelerator backends.

    Attributes:
        CPU: Central Processing Unit backend.
        MPS: Apple Metal Performance Shaders backend.
        CUDA: NVIDIA CUDA backend.
        AUTO: Automatically selects the best available backend.
    """

    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    AUTO = "auto"


class DatasetType(StrEnum):
    """Enumeration of supported federated learning datasets.

    Attributes:
        MNIST: The MNIST dataset of handwritten digits.
        FASHION_MNIST: The Fashion-MNIST dataset of clothing articles.
        CIFAR10: The CIFAR-10 dataset of 10 object classes.
        CIFAR100: The CIFAR-100 dataset of 100 object classes.
    """

    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"

    @property
    def source(self) -> str:
        """Retrieves the source library mapping for the dataset.

        Returns:
            str: The name of the upstream library (e.g., "torchvision").
        """
        match self:
            case (
                DatasetType.MNIST
                | DatasetType.FASHION_MNIST
                | DatasetType.CIFAR10
                | DatasetType.CIFAR100
            ):
                return "torchvision"

    @property
    def source_name(self) -> str:
        """Retrieves the exact dataset class name used by the source library.

        Returns:
            str: The string identifier for the source dataset class.
        """
        match self:
            case DatasetType.MNIST:
                return "MNIST"
            case DatasetType.FASHION_MNIST:
                return "FashionMNIST"
            case DatasetType.CIFAR10:
                return "CIFAR10"
            case DatasetType.CIFAR100:
                return "CIFAR100"

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
    def mean(self) -> tuple[float] | tuple[float, float, float]:
        """Retrieves the channel-wise mean values for dataset normalization.

        Returns:
            tuple[float] | tuple[float, float, float]: A tuple of means for each channel.
        """
        match self:
            case DatasetType.MNIST:
                return (0.1307,)
            case DatasetType.FASHION_MNIST:
                return (0.2860,)
            case DatasetType.CIFAR10:
                return (0.4914, 0.4822, 0.4465)
            case DatasetType.CIFAR100:
                return (0.5071, 0.4865, 0.4409)

    @property
    def std(self) -> tuple[float] | tuple[float, float, float]:
        """Retrieves the channel-wise standard deviation values for dataset normalization.

        Returns:
            tuple[float] | tuple[float, float, float]: A tuple of standard deviations for each channel.
        """
        match self:
            case DatasetType.MNIST:
                return (0.3081,)
            case DatasetType.FASHION_MNIST:
                return (0.3530,)
            case DatasetType.CIFAR10:
                return (0.2470, 0.2435, 0.2616)
            case DatasetType.CIFAR100:
                return (0.2673, 0.2564, 0.2762)

    @property
    def apply_crop_transform(self) -> bool:
        """Determines whether random cropping should be applied during training.

        Returns:
            bool: True if random cropping is enabled, False otherwise.
        """
        match self:
            case DatasetType.CIFAR10 | DatasetType.CIFAR100:
                return True
            case _:
                return False

    @property
    def apply_horizontal_flip_transform(self) -> bool:
        """Determines whether random horizontal flipping should be applied during training.

        Returns:
            bool: True if horizontal flipping is enabled, False otherwise.
        """
        match self:
            case DatasetType.CIFAR10 | DatasetType.CIFAR100:
                return True
            case _:
                return False


class MemoryType(StrEnum):
    """Enumeration of client-side memory tracking strategies.

    Attributes:
        DISABLED: Strategy indicating no memory tracking.
        MODELS: Strategy for tracking historical model weights.
        GRADS: Strategy for tracking historical gradients.
    """

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
