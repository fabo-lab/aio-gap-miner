#!/usr/bin/env python3
"""Remove the leaking features from config, so no script can show an inflated score.

The problem
-----------
`features.build_xy(df)` falls back to `config.FEATURES` when no explicit feature
list is passed. That list still contains `domain_citation_rate` (which is
computed from the label - correlation 1.000 with the domain's observed citation
rate), plus `organic_rank`, `domain_rating` and `page_authority`.

Three scripts call it that way:

    scripts/run_pipeline.py      <- named in the README quickstart
    scripts/build_notebook.py    <- builds the notebook
    scripts/export_tableau.py    <- builds the Tableau data source

Run in that configuration the model scores **PR-AUC 0.93**, with
`domain_citation_rate` among the top features. Anyone who opens the notebook,
the Tableau workbook, or runs the quickstart sees 0.93 instead of 0.35.

What this does
--------------
* comments the leaking entries out of `config.NUMERIC_FEATURES`, with a note
* makes `build_xy` fail loudly if called without explicit feature lists, so the
  fallback can never silently reappear

Both edits are printed before they're applied, and a `.bak` copy is written.

    python scripts/patch_config_leakage.py --dry-run   # show what would change
    python scripts/patch_config_leakage.py
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

LEAKY = ["domain_citation_rate", "domain_rating", "page_authority"]
CONFIG = Path("src/aio_gap_miner/config.py")
FEATURES = Path("src/aio_gap_miner/features.py")


def patch_config(dry: bool) -> bool:
    if not CONFIG.exists():
        print(f"  ! {CONFIG} not found")
        return False
    text = CONFIG.read_text(encoding="utf-8")
    changed = False
    for feat in LEAKY:
        pattern = re.compile(rf'^(\s*)"{feat}",\s*(#.*)?$', re.MULTILINE)
        if pattern.search(text):
            note = ("  # REMOVED: computed from the label / no variance - see the "
                    "leakage audit in the README")
            text = pattern.sub(rf'\1# "{feat}",{note}', text)
            print(f"    comment out  {feat}")
            changed = True
        elif f'"{feat}"' in text:
            print(f"    ? {feat} present but not on its own line - edit by hand")
    if changed and not dry:
        shutil.copy(CONFIG, CONFIG.with_suffix(".py.bak"))
        CONFIG.write_text(text, encoding="utf-8")
    return changed


def patch_features(dry: bool) -> bool:
    if not FEATURES.exists():
        print(f"  ! {FEATURES} not found")
        return False
    text = FEATURES.read_text(encoding="utf-8")
    marker = "numeric_features = config.NUMERIC_FEATURES"
    if marker not in text:
        print("    ? the config fallback in build_xy looks different - check by hand")
        return False
    guard = (
        '        raise ValueError(\n'
        '            "build_xy() was called without explicit feature lists. The old "\n'
        '            "config.FEATURES fallback includes leaking columns and inflates "\n'
        '            "PR-AUC to ~0.93. Pass numeric_features and categorical_features "\n'
        '            "from feature_sets.py instead."\n'
        '        )\n'
    )
    old = ("        numeric_features = config.NUMERIC_FEATURES\n"
           "        categorical_features = config.CATEGORICAL_FEATURES\n")
    if old in text:
        text = text.replace(old, guard)
        print("    build_xy now raises instead of silently using config.FEATURES")
        if not dry:
            shutil.copy(FEATURES, FEATURES.with_suffix(".py.bak"))
            FEATURES.write_text(text, encoding="utf-8")
        return True
    print("    ? the fallback block differs from the expected form - edit by hand")
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print("=" * 74)
    print("  PATCHING THE LEAKY FEATURE FALLBACK" + ("  (dry run)" if args.dry_run else ""))
    print("=" * 74)
    print(f"\n  {CONFIG}:")
    c = patch_config(args.dry_run)
    print(f"\n  {FEATURES}:")
    f = patch_features(args.dry_run)

    print("\n" + "=" * 74)
    if args.dry_run:
        print("  Dry run - nothing written. Re-run without --dry-run to apply.")
    elif c or f:
        print("  Applied. Backups written as *.py.bak")
        print("\n  Now check what still calls the old path:")
        print("    grep -rn 'build_xy(' scripts/ | grep -v numeric_features")
        print("\n  Those scripts will now fail loudly instead of reporting 0.93.")
        print("  Either pass feature_sets lists, or drop them from the README quickstart.")
        print("\n  Then re-run the tests:  pytest -q")
    else:
        print("  Nothing matched - the files may already be patched, or differ.")


if __name__ == "__main__":
    main()
