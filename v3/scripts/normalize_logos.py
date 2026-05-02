#!/usr/bin/env python3
"""
Normalize Kit Sponsors source files into the structured logo_templates layout
expected by the brand registry.

Source layout (the user's original Kit Sponsors folder):
    Kit Sponsors/
    ├── 1 - aon_logo_signature_red_rgb (2).png
    ├── 1 - aon_logo_white_rgb (3).png
    ├── 3 - CCH - Master Logo Black [A3 Digital].png
    ├── 8 - MCP Away.png
    ├── 9 - MCP.png
    ├── ...

Target layout (matches brands.yaml template_path values):
    data/logo_templates/
    ├── brands.yaml
    ├── aon/
    │   ├── red.png
    │   └── white.png
    ├── cch/
    │   ├── black.png
    │   └── white.png
    ├── mcp/
    │   ├── home.png
    │   └── away.png
    ├── chadlaw.png
    ├── em_workwear.png
    └── ...

Usage:
    # Dry-run: print mapping plan
    python scripts/normalize_logos.py --source "../Kit Sponsors/Kit Sponsors" --dry-run

    # Actually copy
    python scripts/normalize_logos.py --source "../Kit Sponsors/Kit Sponsors" --target data/logo_templates

Notes
-----
- For PDF/EPS/SVG sources, we just COPY (you can convert to PNG later via
  inkscape / imagemagick — see comments at the bottom).
- Files that don't match a known mapping are listed at the end so you can
  inspect them and add manual mappings.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from track_annotation.config import load_brand_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Heuristic mapping from source filename → (brand, variant_id)
# ---------------------------------------------------------------------------
# Each entry: a list of substring tokens (case-insensitive) that must ALL
# appear in the filename for the mapping to apply.
# ---------------------------------------------------------------------------

MAPPINGS: list[tuple[list[str], str]] = [
    # (tokens, variant_id from brands.yaml)
    (["aon", "red"], "aon_red"),
    (["aon", "white"], "aon_white"),
    (["atm"], "atm_hospitality"),
    (["cch", "black"], "cch_black"),
    (["cch", "white"], "cch_white"),
    (["chadlaw"], "chadlaw"),
    (["em workwear"], "em_workwear"),
    (["fairway"], "fairway_flooring"),
    (["klg"], "klg"),
    (["mcp away"], "mcp_away"),
    (["mna cladding"], "mna_cladding"),
    (["mna support"], "mna_support"),
    (["top notch"], "top_notch"),
    (["bartercard"], "bartercard"),
    (["floor tonic"], "floor_tonic"),
    (["paints", "yellow"], "paints_lacquers_yellow"),
    (["paints", "red"], "paints_lacquers_red"),
    (["paints", "laquer"], "paints_lacquers_red"),  # fallback if no color word
    (["romantica", "white"], "romantica_white"),
    (["romantica", "black"], "romantica_black"),
    (["acs"], "acs_group"),
    # MCP without "Away" → home (filename like "9 - MCP.png")
    (["mcp"], "mcp_home"),
]


def match_variant(filename: str) -> str | None:
    """Find the first mapping whose tokens all appear in filename."""
    f = filename.lower()
    for tokens, variant_id in MAPPINGS:
        if all(t.lower() in f for t in tokens):
            return variant_id
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path,
                        help='Source folder, e.g. "../Kit Sponsors/Kit Sponsors"')
    parser.add_argument("--target", type=Path, default=ROOT / "data" / "logo_templates",
                        help="Target logo_templates dir")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without copying")
    args = parser.parse_args()

    if not args.source.exists():
        sys.exit(f"Source not found: {args.source}")
    args.target.mkdir(parents=True, exist_ok=True)

    registry_path = args.target / "brands.yaml"
    if not registry_path.exists():
        sys.exit(
            f"brands.yaml not found at {registry_path}. "
            "Create it first (or copy the one shipped with v3)."
        )
    registry = load_brand_registry(registry_path)

    # Build variant -> target_path lookup
    variant_to_path: dict[str, Path] = {}
    for brand in registry.brands:
        for v in brand.variants:
            variant_to_path[v.id] = args.target / v.template_path

    src_files = sorted(p for p in args.source.iterdir() if p.is_file())
    matched: list[tuple[Path, str, Path]] = []
    unmatched: list[Path] = []
    used_variants: set[str] = set()

    for sf in src_files:
        if sf.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"}:
            unmatched.append(sf)
            continue
        variant = match_variant(sf.name)
        if variant is None:
            unmatched.append(sf)
            continue
        if variant in used_variants:
            # Heuristic collision (e.g., "MCP.png" vs "MCP Away.png" both matching "mcp")
            print(f"  ! collision: {sf.name} → {variant} (already taken). Skipping.")
            unmatched.append(sf)
            continue
        if variant not in variant_to_path:
            print(f"  ? mapping says {variant} but registry has no such variant. Skipping.")
            unmatched.append(sf)
            continue
        target = variant_to_path[variant]
        # Preserve source extension if registry path has different ext (e.g., yellow.jpg vs yellow.png)
        if sf.suffix.lower() != target.suffix.lower():
            target = target.with_suffix(sf.suffix.lower())
        matched.append((sf, variant, target))
        used_variants.add(variant)

    # Plan output
    print(f"Plan ({'DRY-RUN' if args.dry_run else 'EXECUTE'}):")
    for sf, variant, tgt in matched:
        print(f"  {sf.name}  →  {variant}  →  {tgt.relative_to(args.target)}")

    if unmatched:
        print(f"\n{len(unmatched)} unmatched source files (please map manually):")
        for sf in unmatched:
            print(f"  - {sf.name}")

    missing_variants = sorted(set(variant_to_path) - used_variants)
    if missing_variants:
        print(f"\n{len(missing_variants)} variants in registry have no source file:")
        for v in missing_variants:
            print(f"  - {v}  →  {variant_to_path[v].relative_to(args.target)}")

    if args.dry_run:
        return

    # Execute
    for sf, _, tgt in matched:
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sf, tgt)
    print(f"\nCopied {len(matched)} files into {args.target}")
    print()
    print("NOTE: For PDF/EPS/SVG sources, you may want to rasterize to PNG, e.g.:")
    print("  inkscape input.svg --export-type=png --export-filename=output.png")
    print("  magick convert input.pdf output.png")


if __name__ == "__main__":
    main()
