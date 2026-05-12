"""Shared analysis utilities for the released alanine-dipeptide examples."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from ggpa.systems.alanine_dipeptide import (
    extract_dimer_torsion_indices,
    extract_monomer_oxygen_indices,
    extract_monomer_torsion_indices,
    extract_valid_strict,
)


ROOT = Path(__file__).resolve().parents[2]


def _public_repo_path(path: str | Path) -> str:
    """Return a repo-relative POSIX path for public-facing metadata."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def compute_dihedral_series(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Vectorised dihedral evaluation for a single 4-atom torsion."""
    pos = np.asarray(positions, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)

    p0 = pos[:, idx[0], :]
    p1 = pos[:, idx[1], :]
    p2 = pos[:, idx[2], :]
    p3 = pos[:, idx[3], :]

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1, axis=1, keepdims=True)
    b1_norm = np.maximum(b1_norm, 1e-12)
    b1 = b1 / b1_norm

    v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1

    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1, v) * w, axis=1)
    return np.arctan2(y, x)


def compute_monomer_dihedrals(positions: np.ndarray, pdb_path: str | Path) -> np.ndarray:
    """Compute (phi, psi) for a monomer trajectory with shape (T, N, 3)."""
    torsion_info = extract_monomer_torsion_indices(str(pdb_path))
    all_idx = np.asarray(torsion_info["all"], dtype=np.int64)
    dihedrals = np.empty((positions.shape[0], all_idx.shape[0]), dtype=np.float64)
    for col, idx in enumerate(all_idx):
        dihedrals[:, col] = compute_dihedral_series(positions, idx)
    return _wrap_pi(dihedrals)


def compute_monomer_oo_distance_nm(positions: np.ndarray, pdb_path: str | Path) -> np.ndarray:
    """Compute the carbonyl O-O distance time series for a monomer trajectory."""
    pair = extract_monomer_oxygen_indices(str(pdb_path))["pair"]
    delta = np.asarray(positions[:, pair[0], :] - positions[:, pair[1], :], dtype=np.float64)
    return np.linalg.norm(delta, axis=1)


def build_reference_cache(
    raw_dir: str | Path,
    pdb_path: str | Path,
    cache_path: str | Path | None = None,
) -> Path:
    """Aggregate all `traj_all_*.npy` files in a directory into one dihedral cache."""
    raw_dir = Path(raw_dir)
    pdb_path = Path(pdb_path)
    if cache_path is None:
        cache_path = raw_dir / "ramachandran_deg.npz"
    cache_path = Path(cache_path)

    traj_paths = sorted(raw_dir.glob("traj_all_*.npy"))
    if not traj_paths:
        raise FileNotFoundError(f"No traj_all_*.npy files found in {raw_dir}")

    chunks = []
    for traj_path in traj_paths:
        coords = np.load(traj_path)
        chunks.append(compute_monomer_dihedrals(coords, pdb_path))
    dihedrals_rad = np.concatenate(chunks, axis=0)
    dihedrals_deg = np.degrees(dihedrals_rad)

    np.savez_compressed(
        cache_path,
        dihedrals_rad=dihedrals_rad,
        dihedrals_deg=dihedrals_deg,
        n_frames=np.array(len(dihedrals_rad), dtype=np.int64),
        n_files=np.array(len(traj_paths), dtype=np.int64),
    )
    return cache_path


def save_ad_sodium_reference_npz(
    output_path: str | Path,
    dihedrals_rad: np.ndarray,
    oo_distance_nm: np.ndarray,
    source_label: str,
    topology_pdb: str | Path,
) -> Path:
    """Save one curated AD-Na+ comparison dataset with a unified schema."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    oxygen_meta = extract_monomer_oxygen_indices(str(topology_pdb))
    np.savez_compressed(
        output_path,
        dihedrals_rad=np.asarray(dihedrals_rad, dtype=np.float64),
        dihedrals_deg=np.degrees(np.asarray(dihedrals_rad, dtype=np.float64)),
        oo_distance_nm=np.asarray(oo_distance_nm, dtype=np.float64),
        n_frames=np.array(len(dihedrals_rad), dtype=np.int64),
        source_label=np.array(source_label),
        topology_pdb=np.array(_public_repo_path(topology_pdb)),
        oxygen_pair=np.asarray(oxygen_meta["pair"], dtype=np.int64),
        oxygen_atom_names=np.asarray(oxygen_meta["atom_names"]),
        oxygen_residue_names=np.asarray(oxygen_meta["residue_names"]),
    )
    return output_path


def load_reference_dihedrals(
    raw_dir: str | Path,
    pdb_path: str | Path,
    cache_path: str | Path | None = None,
) -> Dict[str, np.ndarray]:
    """Load cached monomer reference dihedrals, building the cache if needed."""
    raw_dir = Path(raw_dir)
    if cache_path is None:
        cache_path = raw_dir / "ramachandran_deg.npz"
    cache_path = Path(cache_path)

    if not cache_path.exists():
        build_reference_cache(raw_dir, pdb_path, cache_path)

    payload = np.load(cache_path)
    return {
        "dihedrals_rad": np.asarray(payload["dihedrals_rad"], dtype=np.float64),
        "dihedrals_deg": np.asarray(payload["dihedrals_deg"], dtype=np.float64),
        "n_frames": int(payload["n_frames"]),
        "n_files": int(payload["n_files"]),
        "cache_path": cache_path,
    }


def load_curated_ad_sodium_reference(path: str | Path) -> Dict[str, np.ndarray]:
    """Load one curated AD-Na+ comparison dataset from `data/ad_sodium_ref`."""
    payload = np.load(path, allow_pickle=True)
    return {
        "dihedrals_rad": np.asarray(payload["dihedrals_rad"], dtype=np.float64),
        "dihedrals_deg": np.asarray(payload["dihedrals_deg"], dtype=np.float64),
        "oo_distance_nm": np.asarray(payload["oo_distance_nm"], dtype=np.float64),
        "n_frames": int(payload["n_frames"]),
        "source_label": str(payload["source_label"]),
        "topology_pdb": str(payload["topology_pdb"]),
        "oxygen_pair": np.asarray(payload["oxygen_pair"], dtype=np.int64),
    }


def save_ad_dimer_reference_npz(
    output_path: str | Path,
    analysis: Dict[str, np.ndarray],
    source_label: str,
    topology_pdb: str | Path,
    trajectory_dcd: str | Path,
) -> Path:
    """Save one curated AD dimer reference dataset with a unified schema."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        com_dist_nm=np.asarray(analysis["com_distances"], dtype=np.float64),
        cosine_sim=np.asarray(analysis["cosine_similarities"], dtype=np.float64),
        hbond_counts=np.asarray(analysis["hbond_counts"], dtype=np.float64),
        reciprocal_counts=np.asarray(analysis["reciprocal_counts"], dtype=np.float64),
        dihedrals_seg1_rad=np.asarray(analysis["dihedrals_seg1"], dtype=np.float64),
        dihedrals_seg1_deg=np.degrees(np.asarray(analysis["dihedrals_seg1"], dtype=np.float64)),
        dihedrals_seg2_rad=np.asarray(analysis["dihedrals_seg2"], dtype=np.float64),
        dihedrals_seg2_deg=np.degrees(np.asarray(analysis["dihedrals_seg2"], dtype=np.float64)),
        n_frames=np.array(len(analysis["com_distances"]), dtype=np.int64),
        source_label=np.array(source_label),
        topology_pdb=np.array(_public_repo_path(topology_pdb)),
        trajectory_dcd=np.array(
            _public_repo_path(trajectory_dcd) if Path(trajectory_dcd).exists() else "not_shipped"
        ),
    )
    return output_path


def build_ad_dimer_reference_npz(
    dcd_path: str | Path,
    pdb_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Process a dimer MD trajectory and save a curated reference NPZ.

    This path intentionally avoids the slower per-frame Python loop used by the
    generic analysis helper. The dimer reference is large enough that a mostly
    vectorized implementation is noticeably faster and more robust than naive
    multiprocessing.
    """
    import mdtraj as md

    dcd_path = Path(dcd_path)
    pdb_path = Path(pdb_path)
    traj = md.load_dcd(str(dcd_path), top=str(pdb_path))
    xyz = np.asarray(traj.xyz, dtype=np.float64)

    torsion_info = extract_dimer_torsion_indices(str(pdb_path))
    chain_atoms = torsion_info["chain_atom_lists"]
    n_atoms = torsion_info["n_atoms"]
    mid = n_atoms // 2

    seg1 = traj.atom_slice(range(mid))
    seg2 = traj.atom_slice(range(mid, n_atoms))

    com1 = md.compute_center_of_mass(seg1)
    com2 = md.compute_center_of_mass(seg2)
    com_dist = np.linalg.norm(com1 - com2, axis=1)

    phi1 = md.compute_phi(seg1)[1]
    psi1 = md.compute_psi(seg1)[1]
    phi2 = md.compute_phi(seg2)[1]
    psi2 = md.compute_psi(seg2)[1]
    dihedrals_seg1 = np.hstack([phi1, psi1]).astype(np.float64, copy=False)
    dihedrals_seg2 = np.hstack([phi2, psi2]).astype(np.float64, copy=False)

    top = md.load_topology(str(pdb_path))

    def _orientation_endpoints(atom_indices):
        sub_top = top.subset(atom_indices)
        c_atoms = sub_top.select("name C")
        n_atoms_sel = sub_top.select("name N")
        if len(c_atoms) > 0 and len(n_atoms_sel) > 0:
            return atom_indices[c_atoms[0]], atom_indices[n_atoms_sel[-1]]
        return atom_indices[0], atom_indices[-1]

    s0, e0 = _orientation_endpoints(chain_atoms[0])
    s1, e1 = _orientation_endpoints(chain_atoms[1])
    vec0 = xyz[:, e0, :] - xyz[:, s0, :]
    vec1 = xyz[:, e1, :] - xyz[:, s1, :]
    dot = np.sum(vec0 * vec1, axis=1)
    cosine_sim = dot / (np.linalg.norm(vec0, axis=1) * np.linalg.norm(vec1, axis=1) + 1e-12)

    all_hb = md.baker_hubbard(traj, periodic=False)
    inter = np.array([b for b in all_hb if (b[0] < mid) != (b[2] < mid)])
    hbond_counts = np.zeros(traj.n_frames, dtype=np.float64)
    reciprocal_counts = np.zeros(traj.n_frames, dtype=np.float64)
    if len(inter) > 0:
        dists = md.compute_distances(traj, inter[:, [0, 2]])
        angles = md.compute_angles(traj, inter)
        is_hb = (dists < 0.35) & (angles > np.radians(120))
        hbond_counts = np.sum(is_hb, axis=1).astype(np.float64)
        for f in range(traj.n_frames):
            active = inter[is_hb[f]]
            if len(active) >= 2:
                donors = active[:, 0]
                if np.any(donors < mid) and np.any(donors >= mid):
                    reciprocal_counts[f] = 1.0

    analysis = {
        "com_distances": com_dist,
        "cosine_similarities": cosine_sim,
        "hbond_counts": hbond_counts,
        "reciprocal_counts": reciprocal_counts,
        "dihedrals_seg1": dihedrals_seg1,
        "dihedrals_seg2": dihedrals_seg2,
    }
    return save_ad_dimer_reference_npz(
        output_path=output_path,
        analysis=analysis,
        source_label="AD dimer MD reference (100 ns, 1 traj)",
        topology_pdb=pdb_path,
        trajectory_dcd=dcd_path,
    )


def load_curated_ad_dimer_reference(path: str | Path) -> Dict[str, np.ndarray]:
    """Load one curated AD dimer reference dataset from `data/ad_dimer_ref`."""
    payload = np.load(path, allow_pickle=True)
    return {
        "com_dist": np.asarray(payload["com_dist_nm"], dtype=np.float64),
        "cosine_sim": np.asarray(payload["cosine_sim"], dtype=np.float64),
        "hbond_counts": np.asarray(payload["hbond_counts"], dtype=np.float64),
        "reciprocal_counts": np.asarray(payload["reciprocal_counts"], dtype=np.float64),
        "dihedrals_seg1": np.asarray(payload["dihedrals_seg1_rad"], dtype=np.float64),
        "dihedrals_seg2": np.asarray(payload["dihedrals_seg2_rad"], dtype=np.float64),
        "n_frames": int(payload["n_frames"]),
        "source_label": str(payload["source_label"]),
        "topology_pdb": str(payload["topology_pdb"]),
        "trajectory_dcd": str(payload["trajectory_dcd"]),
    }


def burnin_slice(n_frames: int, burnin_fraction: float) -> slice:
    """Return a post-burnin slice for a trajectory of length `n_frames`."""
    burnin = int(np.floor(float(burnin_fraction) * int(n_frames)))
    burnin = max(0, min(burnin, int(n_frames)))
    return slice(burnin, None)


def load_ad_sodium_result(result_path: str | Path) -> Dict[str, np.ndarray]:
    payload = np.load(result_path)
    return {
        "steps": np.asarray(payload["steps"], dtype=np.int64),
        "y_deg": np.asarray(payload["dihedrals_deg"], dtype=np.float64),
        "x_deg": np.asarray(payload["x_dihedrals_deg"], dtype=np.float64),
        "y_rad": np.asarray(payload["dihedrals_rad"], dtype=np.float64),
        "x_rad": np.asarray(payload["x_dihedrals_rad"], dtype=np.float64),
        "oo_distance_nm": np.asarray(payload["oo_distance_nm"], dtype=np.float64),
        "oxygen_pair": np.asarray(payload["oxygen_pair"], dtype=np.int64),
        "final_positions_nm": np.asarray(payload["final_positions_nm"], dtype=np.float64),
        "t_diff": float(payload["t_diff"]),
        "wall_time_s": float(payload["wall_time_s"]),
    }


def load_ad_dimer_result(result_path: str | Path) -> Dict[str, np.ndarray]:
    payload = np.load(result_path)
    return {
        "production_steps": np.asarray(payload["production_steps"], dtype=np.int64),
        "production_dihedrals_deg": np.asarray(payload["production_dihedrals_deg"], dtype=np.float64),
        "production_x_dihedrals_deg": np.asarray(payload["production_x_dihedrals_deg"], dtype=np.float64),
        "analysis_dihedrals_seg1": np.asarray(payload["analysis_dihedrals_seg1"], dtype=np.float64),
        "analysis_dihedrals_seg2": np.asarray(payload["analysis_dihedrals_seg2"], dtype=np.float64),
        "analysis_cosine_similarities": np.asarray(payload["analysis_cosine_similarities"], dtype=np.float64),
        "analysis_hbond_counts": np.asarray(payload["analysis_hbond_counts"], dtype=np.float64),
        "analysis_reciprocal_counts": np.asarray(payload["analysis_reciprocal_counts"], dtype=np.float64),
        "wall_time_s": float(payload["wall_time_s"]),
        "t_diffs": np.asarray(payload["t_diffs"], dtype=np.float64),
    }


def strict_dimer_masks(dimer_payload: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return strict antiparallel/parallel masks using the released criteria."""
    res = {
        "cosine_similarities": dimer_payload["analysis_cosine_similarities"],
        "hbond_counts": dimer_payload["analysis_hbond_counts"],
        "reciprocal_counts": dimer_payload["analysis_reciprocal_counts"],
        "dihedrals_seg1": dimer_payload["analysis_dihedrals_seg1"],
        "dihedrals_seg2": dimer_payload["analysis_dihedrals_seg2"],
    }
    return extract_valid_strict(res)


def wrapped_abs_angle_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Absolute angular difference folded into [0, 180] degrees."""
    a = np.deg2rad(np.asarray(a_deg, dtype=np.float64))
    b = np.deg2rad(np.asarray(b_deg, dtype=np.float64))
    diff = np.abs(a - b)
    diff = np.minimum(diff, 2.0 * np.pi - diff)
    return np.degrees(diff)
