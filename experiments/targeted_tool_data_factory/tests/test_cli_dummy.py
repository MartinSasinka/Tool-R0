"""Generic dummy target adapter + CLI smoke + resume/cache guard."""
import shutil
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from targeted_tool_data.cli import main            # noqa: E402
from targeted_tool_data.util import StepGuard      # noqa: E402

VERSION = "vtestdummy"


def _cleanup():
    out = MODULE_ROOT / "outputs"
    for sub in out.glob("**/*"):
        pass
    for sub in ("profiles", "candidates", "validated", "selected", "splits",
                "reports"):
        d = out / sub
        if d.exists():
            for f in list(d.glob(f"*{VERSION}*")) + list(d.glob("dummy_*")) \
                    + list(d.glob("DUMMY_*")) + list(d.glob(f"*_profile_{VERSION}*")):
                if f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    f.unlink(missing_ok=True)


@pytest.fixture(scope="module", autouse=True)
def clean():
    _cleanup()
    yield
    _cleanup()


def test_cli_all_on_dummy_target():
    """Pipeline runs end-to-end on a non-NESTFUL target (genericity §20)."""
    main(["all", "--target", "dummy", "--version", VERSION,
          "--candidates", "40", "--dry-run", "--overwrite",
          "--seed", "7", "--no-llm", "--no-docs"])
    out = MODULE_ROOT / "outputs"
    assert (out / "profiles" / "dummy_profile.json").is_file()
    assert (out / "candidates" / f"candidates_{VERSION}.jsonl").is_file()
    assert (out / "validated" / f"validated_{VERSION}.jsonl").is_file()
    assert (out / "selected" / f"selected_{VERSION}.jsonl").is_file()
    assert (out / "splits" / f"leakage_audit_{VERSION}.json").is_file()
    import json
    audit = json.loads((out / "splits" / f"leakage_audit_{VERSION}.json")
                       .read_text(encoding="utf-8"))
    assert not audit["leaked"]
    manifest_files = list((out / "selected" / f"export_{VERSION}")
                          .glob("manifest_*.json"))
    assert manifest_files


def test_step_guard_resume_and_overwrite(tmp_path):
    g = StepGuard(tmp_path, "s1", resume=False, overwrite=False)
    assert not g.should_skip()
    g.mark({"x": 1})
    g2 = StepGuard(tmp_path, "s1", resume=True, overwrite=False)
    assert g2.should_skip()        # resume skips completed step
    g3 = StepGuard(tmp_path, "s1", resume=False, overwrite=False)
    with pytest.raises(SystemExit):
        g3.should_skip()           # overwrite protection
    g4 = StepGuard(tmp_path, "s1", resume=False, overwrite=True)
    assert not g4.should_skip()
