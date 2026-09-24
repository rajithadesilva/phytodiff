from __future__ import annotations

import pytest

from tomato_recon.infer import main


def test_legacy_multistage_inference_reports_pending_integration() -> None:
    with pytest.raises(RuntimeError, match="await DiffTS"):
        main([])
