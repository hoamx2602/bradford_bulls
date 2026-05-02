#!/usr/bin/env python3
"""
Validate that the v3 environment is correctly set up.

Checks:
  1. Python version
  2. Required packages can be imported
  3. CUDA / MPS availability
  4. YOLO weights exist and load
  5. Config files exist
  6. Logo templates directory exists
  7. Disk write access to data dirs

Usage:
    python scripts/validate_setup.py
    python scripts/validate_setup.py --weights weights/yolo11l.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def check(name: str, ok: bool, detail: str = ""):
    status = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
    line = f"[{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolo11l.pt")
    args = parser.parse_args()

    failures = 0

    # 1. Python version
    py_ok = sys.version_info >= (3, 10)
    if not check("Python >= 3.10", py_ok, f"current: {sys.version.split()[0]}"):
        failures += 1

    # 2. Imports
    try:
        import torch  # noqa: F401
        check("import torch", True, f"version {torch.__version__}")
    except ImportError as e:
        check("import torch", False, str(e))
        failures += 1

    try:
        import ultralytics  # noqa: F401
        check("import ultralytics", True, f"version {ultralytics.__version__}")
    except ImportError as e:
        check("import ultralytics", False, str(e))
        failures += 1

    try:
        import cv2  # noqa: F401
        check("import cv2", True, f"version {cv2.__version__}")
    except ImportError as e:
        check("import cv2", False, str(e))
        failures += 1

    try:
        import streamlit  # noqa: F401
        check("import streamlit", True, f"version {streamlit.__version__}")
    except ImportError as e:
        check("import streamlit", False, str(e))
        failures += 1

    try:
        from track_annotation.config import load_config  # noqa: F401
        check("import track_annotation", True)
    except ImportError as e:
        check("import track_annotation", False, str(e))
        failures += 1

    # 3. GPU
    try:
        import torch
        if torch.cuda.is_available():
            check(
                "CUDA available",
                True,
                f"{torch.cuda.device_count()} device(s), {torch.cuda.get_device_name(0)}",
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            check("MPS available", True, "Apple Silicon")
        else:
            print(f"[{YELLOW}WARN{RESET}] No GPU detected; will use CPU (slow)")
    except Exception as e:  # noqa: BLE001
        check("GPU check", False, str(e))
        failures += 1

    # 4. Weights
    if args.weights.exists():
        size_mb = args.weights.stat().st_size / 1024 / 1024
        check(f"weights {args.weights.name}", True, f"{size_mb:.1f} MB")
        try:
            from ultralytics import YOLO
            _ = YOLO(str(args.weights))
            check("YOLO weights loadable", True)
        except Exception as e:  # noqa: BLE001
            check("YOLO weights loadable", False, str(e))
            failures += 1
    else:
        check(f"weights {args.weights}", False, "file not found")
        print(
            f"  -> Copy your weights here: cp /path/to/yolo11l.pt {args.weights}"
        )
        failures += 1

    # 5. Config files
    for cfg_name in ("default.yaml", "person_tracking.yaml", "logo_tracking.yaml"):
        p = ROOT / "configs" / cfg_name
        check(f"config {cfg_name}", p.exists())
        if not p.exists():
            failures += 1

    # 6. Logo templates + brand registry
    templates_dir = ROOT / "data" / "logo_templates"
    if templates_dir.exists():
        n = len(
            list(templates_dir.glob("**/*.png"))
            + list(templates_dir.glob("**/*.jpg"))
            + list(templates_dir.glob("**/*.jpeg"))
        )
        check("logo_templates/", True, f"{n} image files")
    else:
        check("logo_templates/", False, "dir missing")
        failures += 1

    registry_path = templates_dir / "brands.yaml"
    if registry_path.exists():
        try:
            from track_annotation.config import load_brand_registry
            reg = load_brand_registry(registry_path)
            n_variants = sum(len(b.variants) for b in reg.brands)
            check(
                "brands.yaml registry",
                True,
                f"{len(reg.brands)} brands, {n_variants} variants",
            )

            # Verify each variant template path exists
            missing = []
            for b in reg.brands:
                for v in b.variants:
                    if not (templates_dir / v.template_path).exists():
                        missing.append(f"{v.id} → {v.template_path}")
            if missing:
                print(
                    f"[{YELLOW}WARN{RESET}] {len(missing)} template files missing "
                    f"(run scripts/normalize_logos.py to populate):"
                )
                for m in missing[:10]:
                    print(f"  - {m}")
                if len(missing) > 10:
                    print(f"  ... and {len(missing) - 10} more")
        except Exception as e:  # noqa: BLE001
            check("brands.yaml parse", False, str(e))
            failures += 1
    else:
        check("brands.yaml", False, "missing — see data/logo_templates/brands.yaml")
        failures += 1

    # 7. Write access
    for d in ("data/annotation_packages", "logs"):
        p = ROOT / d
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".write_test"
            test_file.write_text("ok")
            test_file.unlink()
            check(f"write access {d}/", True)
        except Exception as e:  # noqa: BLE001
            check(f"write access {d}/", False, str(e))
            failures += 1

    print()
    if failures == 0:
        print(f"{GREEN}All checks passed.{RESET} You can run:")
        print(f"  python scripts/run_pipeline.py --video VIDEO --config configs/person_tracking.yaml --output OUT")
        return 0
    else:
        print(f"{RED}{failures} check(s) failed.{RESET} See messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
