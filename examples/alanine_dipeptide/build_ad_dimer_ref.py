#!/usr/bin/env python3
"""Build a curated AD dimer reference package from the local MD trajectory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from analysis_utils import build_ad_dimer_reference_npz, load_curated_ad_dimer_reference


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ref_src = root / "data/md_dimer_100ns_1traj"
    ref_dir = root / "data/ad_dimer_ref"
    ref_dir.mkdir(parents=True, exist_ok=True)

    dcd_path = ref_src / "trajectory_al2dimer_0.dcd"
    pdb_src_path = ref_src / "final_structure_al2dimer_0.pdb"
    pdb_packaged_path = ref_dir / "ad_dimer_md_100ns_ref_topology.pdb"
    output_path = ref_dir / "ad_dimer_md_100ns_ref.npz"
    manifest_path = ref_dir / "manifest.json"

    shutil.copy2(pdb_src_path, pdb_packaged_path)

    build_ad_dimer_reference_npz(dcd_path=dcd_path, pdb_path=pdb_packaged_path, output_path=output_path)
    payload = load_curated_ad_dimer_reference(output_path)

    manifest = {
        "datasets": [
            {
                "name": "ad_dimer_md_100ns_ref",
                "path": str(output_path),
                "source_label": payload["source_label"],
                "n_frames": payload["n_frames"],
                "topology_pdb": payload["topology_pdb"],
                "source_trajectory_dcd": str(dcd_path),
                "fields": [
                    "com_dist",
                    "cosine_sim",
                    "hbond_counts",
                    "reciprocal_counts",
                    "dihedrals_seg1",
                    "dihedrals_seg2",
                ],
            }
        ]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
