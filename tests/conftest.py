import os
import tempfile
from pathlib import Path

import pytest

from bigmoe.errors import BoxError
from bigmoe.system import check_odirect


@pytest.fixture
def odirect_dir(tmp_path):
    """tmp_path may be on tmpfs, which has no O_DIRECT: fall back to $BIGMOE_TEST_DIR or ~/.cache."""
    for base in (tmp_path, os.environ.get("BIGMOE_TEST_DIR"), Path.home() / ".cache"):
        if not base:
            continue
        Path(base).mkdir(parents=True, exist_ok=True)
        d = Path(tempfile.mkdtemp(prefix="bigmoe-test-", dir=base))
        try:
            check_odirect(d)
            yield d
            import shutil
            shutil.rmtree(d, ignore_errors=True)
            return
        except BoxError:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
    pytest.skip("no O_DIRECT-capable directory")
