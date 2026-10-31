import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """
    Enforces LLM_PROVIDER=mock during automated pytest execution
    so tests run offline and deterministically without requiring external API keys.
    """
    os.environ["LLM_PROVIDER"] = "mock"
    from app.config import settings
    settings.LLM_PROVIDER = "mock"
