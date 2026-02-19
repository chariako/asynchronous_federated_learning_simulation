import pytest
from loguru import logger


@pytest.fixture
def capture_logs(caplog):
    """
    Global fixture to capture loguru logs using pytest's standard caplog.
    """
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)
