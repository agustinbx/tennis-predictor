"""
Fixtures compartidas para tests.
"""
import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Fixture que devuelve el directorio raíz del proyecto."""
    return Path(__file__).parent.parent
