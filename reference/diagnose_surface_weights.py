#!/usr/bin/env python3
"""Isolate the mesh-only portion of upstream voxel_skin for parity work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.fixture / "animation-vertices.json").read_text())
    n, j, f = manifest["vertex_count"], manifest["joint_count"], manifest["face_count"]
    vertices = np.fromfile(args.fixture / "normalized-vertices.f32", "<f4").reshape(n, 3)
    joints = np.fromfile(args.fixture / "normalized-joints.f32", "<f4").reshape(j, 3)
    faces = np.fromfile(args.fixture / "faces.u32", "<u4").reshape(f, 3)

    vertex_tree = cKDTree(vertices)
    close_distance, close_index = vertex_tree.query(vertices, 4)
    close_distance, close_index = close_distance[:, 1:], close_index[:, 1:]
    close_mask = (close_distance > 0) & (close_distance < 1.0e-5)
    close_source = np.repeat(np.arange(n), 3)[close_mask.ravel()]
    close_target = close_index[close_mask]
    close_weight = close_distance[close_mask]
    source = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2], close_source))
    target = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0], close_target))
    edge_weight = np.sqrt(((vertices[source] - vertices[target]) ** 2).sum(axis=-1))
    if len(close_weight):
        edge_weight[-len(close_weight):] = close_weight
    graph = csr_matrix((edge_weight, (source, target)), shape=(n, n))
    _, seeds = vertex_tree.query(joints)
    distance = shortest_path(graph, method="D", directed=False, indices=seeds)
    unreachable = np.isinf(distance).all(axis=0)
    nearest_distance, nearest_joint = cKDTree(joints).query(vertices[unreachable], min(j, 3))
    unreachable_indices = np.flatnonzero(unreachable)
    distance[nearest_joint, np.repeat(unreachable_indices, min(j, 3)).reshape(-1, min(j, 3))] = nearest_distance
    maximum = np.max(distance[np.isfinite(distance)])
    distance = np.nan_to_num(distance, nan=maximum, posinf=maximum, neginf=maximum)
    distance = np.maximum(distance, 1.0e-6)
    weights = (1.0 / (0.5 * distance + 0.5 * distance**2))**2
    weights = (weights / weights.sum(axis=0)).T.astype("<f4")
    official = np.fromfile(args.fixture / "reference-surface-weights.f32", "<f4").reshape(n, j)
    official_seeds = np.fromfile(args.fixture / "reference-surface-seeds.u32", "<u4")
    seed_ties = []
    for joint, point in enumerate(joints):
        squared = ((vertices - point) ** 2).sum(axis=1)
        candidates = np.flatnonzero(squared == squared.min())
        seed_ties.append({"joint": joint, "official": int(official_seeds[joint]),
                          "candidates": candidates.tolist()})
    delta = weights.astype(np.float64) - official.astype(np.float64)
    pairs = np.unique(np.sort(np.stack((source, target), axis=1), axis=1), axis=0)
    pair_weight = np.sqrt(((vertices[pairs[:, 0]] - vertices[pairs[:, 1]]) ** 2).sum(axis=-1))
    unique_graph = csr_matrix((np.concatenate((pair_weight, pair_weight)),
                               (np.concatenate((pairs[:, 0], pairs[:, 1])),
                                np.concatenate((pairs[:, 1], pairs[:, 0])))), shape=(n, n))
    unique_distance = shortest_path(unique_graph, method="D", directed=True, indices=seeds)
    unique_unreachable = np.isinf(unique_distance).all(axis=0)
    ud, uj = cKDTree(joints).query(vertices[unique_unreachable], min(j, 3))
    ui = np.flatnonzero(unique_unreachable)
    unique_distance[uj, np.repeat(ui, min(j, 3)).reshape(-1, min(j, 3))] = ud
    unique_maximum = np.max(unique_distance[np.isfinite(unique_distance)])
    unique_distance = np.maximum(np.nan_to_num(
        unique_distance, nan=unique_maximum, posinf=unique_maximum, neginf=unique_maximum), 1e-6)
    unique_weights = (1.0 / (0.5 * unique_distance + 0.5 * unique_distance**2))**2
    unique_weights = (unique_weights / unique_weights.sum(axis=0)).T.astype("<f4")
    unique_delta = unique_weights.astype(np.float64) - weights.astype(np.float64)
    unique_official = unique_weights.astype(np.float64) - official.astype(np.float64)
    print(json.dumps({
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta.ravel()) / np.linalg.norm(official.ravel())),
        "close_edges": int(len(close_weight)),
        "unreachable_vertices": int(unreachable.sum()),
        "unique_vs_scipy_relative_l2": float(np.linalg.norm(unique_delta.ravel()) / np.linalg.norm(weights.ravel())),
        "unique_vs_official_relative_l2": float(np.linalg.norm(unique_official.ravel()) / np.linalg.norm(official.ravel())),
        "unique_unreachable_vertices": int(unique_unreachable.sum()),
        "seed_ties": seed_ties,
    }, indent=2))
    weights.tofile(args.fixture / "reference-mesh-surface-weights.f32")


if __name__ == "__main__":
    main()
