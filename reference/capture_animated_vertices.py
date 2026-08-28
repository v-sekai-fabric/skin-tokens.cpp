#!/usr/bin/env python3
"""Capture final animated vertices from a SkinTokens GLB with NumPy LBS.

This intentionally does not call the C++ runtime. It reads the glTF hierarchy,
animation, inverse bind matrices and four skin influences, then applies glTF
linear-blend skinning at exact keyframes. The resulting fixture is consumed by
skintokens-animation-parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
DTYPES = {5121: np.dtype("u1"), 5123: np.dtype("<u2"), 5125: np.dtype("<u4"),
          5126: np.dtype("<f4")}


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    if len(data) < 20 or struct.unpack_from("<II", data) != (0x46546C67, 2):
        raise ValueError("input is not a glTF 2 GLB")
    declared = struct.unpack_from("<I", data, 8)[0]
    if declared != len(data):
        raise ValueError("GLB length does not match its header")
    root = None
    binary = b""
    offset = 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + length]
        if len(chunk) != length:
            raise ValueError("truncated GLB chunk")
        offset += length
        if kind == 0x4E4F534A:
            root = json.loads(chunk.decode("utf-8"))
        elif kind == 0x004E4942:
            binary = chunk
    if root is None:
        raise ValueError("GLB has no JSON chunk")
    return root, binary


def accessor(root: dict, binary: bytes, index: int) -> np.ndarray:
    item = root["accessors"][index]
    if "sparse" in item:
        raise ValueError("sparse accessors are not supported by this parity capture")
    view = root["bufferViews"][item["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise ValueError("external buffers are not valid in this GLB capture")
    dtype = DTYPES[item["componentType"]]
    components = COMPONENTS[item["type"]]
    count = item["count"]
    offset = view.get("byteOffset", 0) + item.get("byteOffset", 0)
    packed = dtype.itemsize * components
    stride = view.get("byteStride", packed)
    required = offset + (count - 1) * stride + packed if count else offset
    if required > len(binary):
        raise ValueError("accessor exceeds the GLB binary chunk")
    value = np.ndarray((count, components), dtype=dtype, buffer=binary, offset=offset,
                       strides=(stride, dtype.itemsize)).copy()
    return value[:, 0] if components == 1 else value


def quaternion_matrix(value: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(component) for component in value)
    length = x*x + y*y + z*z + w*w
    scale = 2.0 / length if length > 0.0 else 0.0
    xx, yy, zz = x*x*scale, y*y*scale, z*z*scale
    xy, xz, yz = x*y*scale, x*z*scale, y*z*scale
    wx, wy, wz = w*x*scale, w*y*scale, w*z*scale
    return np.array([[1-yy-zz, xy-wz, xz+wy, 0],
                     [xy+wz, 1-xx-zz, yz-wx, 0],
                     [xz-wy, yz+wx, 1-xx-yy, 0],
                     [0, 0, 0, 1]], dtype=np.float64)


def local_matrix(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    output = quaternion_matrix(rotation)
    output[:3, 3] = translation
    return output


def selected_frames(text: str | None, count: int) -> list[int]:
    if text:
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    else:
        values = [0, round((count - 1) * .25), round((count - 1) * .5),
                  round((count - 1) * .75), count - 1]
    values = list(dict.fromkeys(values))
    if not values or min(values) < 0 or max(values) >= count:
        raise ValueError("selected frame is outside the animation")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", help="comma-separated exact keyframe indices")
    args = parser.parse_args()
    root, binary = read_glb(args.glb)
    if len(root.get("skins", [])) != 1 or not root.get("animations"):
        raise ValueError("GLB must contain one skin and an animation")
    skin = root["skins"][0]
    joint_nodes = skin["joints"]
    joint_count = len(joint_nodes)
    node_to_joint = {node: joint for joint, node in enumerate(joint_nodes)}
    parents = np.full(joint_count, -1, dtype="<i4")
    for parent_node, node in enumerate(root["nodes"]):
        for child_node in node.get("children", []):
            if parent_node in node_to_joint and child_node in node_to_joint:
                parents[node_to_joint[child_node]] = node_to_joint[parent_node]
    roots = np.flatnonzero(parents < 0)
    if roots.tolist() != [0]:
        raise ValueError("the C++ exact-frame evaluator requires joint zero to be the sole root")
    for joint, parent in enumerate(parents):
        if parent >= joint:
            raise ValueError("skin joints are not in parent-before-child order")

    mesh_nodes = [node for node in root["nodes"] if "mesh" in node and "skin" in node]
    if len(mesh_nodes) != 1:
        raise ValueError("GLB must contain one skinned mesh node")
    primitive = root["meshes"][mesh_nodes[0]["mesh"]]["primitives"][0]
    attributes = primitive["attributes"]
    positions = accessor(root, binary, attributes["POSITION"]).astype("<f4")
    joints = accessor(root, binary, attributes["JOINTS_0"]).astype("<u2")
    weights = accessor(root, binary, attributes["WEIGHTS_0"]).astype("<f4")
    if "indices" not in primitive:
        raise ValueError("parity capture requires an indexed triangle mesh")
    faces = accessor(root, binary, primitive["indices"]).astype("<u4").reshape(-1, 3)
    if positions.shape[1:] != (3,) or joints.shape != (len(positions), 4) or weights.shape != joints.shape:
        raise ValueError("unexpected skinned vertex stream shape")
    inverse_bind = accessor(root, binary, skin["inverseBindMatrices"]).astype(np.float64)
    inverse_bind = inverse_bind.reshape(joint_count, 4, 4).transpose(0, 2, 1)
    rest_positions = np.stack([np.linalg.inv(value)[:3, 3] for value in inverse_bind]).astype("<f4")

    node_translations = np.zeros((joint_count, 3), dtype=np.float64)
    node_rotations = np.zeros((joint_count, 4), dtype=np.float64)
    node_rotations[:, 3] = 1.0
    for joint, node_index in enumerate(joint_nodes):
        node = root["nodes"][node_index]
        if "matrix" in node or "scale" in node:
            raise ValueError("matrix/scale joint nodes are outside the exporter parity contract")
        node_translations[joint] = node.get("translation", [0, 0, 0])
        node_rotations[joint] = node.get("rotation", [0, 0, 0, 1])

    animation = root["animations"][0]
    samplers = animation["samplers"]
    channels = animation["channels"]
    time_arrays = [accessor(root, binary, sampler["input"]).astype(np.float64) for sampler in samplers]
    frame_count = max(len(value) for value in time_arrays)
    if frame_count < 1:
        raise ValueError("animation contains no keyframes")
    local_rotations = np.broadcast_to(node_rotations, (frame_count, joint_count, 4)).copy()
    root_translations = np.broadcast_to(node_translations[0], (frame_count, 3)).copy()
    common_times = None
    for channel in channels:
        sampler = samplers[channel["sampler"]]
        times = accessor(root, binary, sampler["input"]).astype(np.float64)
        values = accessor(root, binary, sampler["output"]).astype(np.float64)
        if len(times) != frame_count or len(values) != frame_count:
            raise ValueError("parity capture requires frame-aligned animation tracks")
        if common_times is None:
            common_times = times
        elif not np.array_equal(common_times, times):
            raise ValueError("parity capture requires a common keyframe timeline")
        target = channel["target"]
        if target["node"] not in node_to_joint:
            continue
        joint = node_to_joint[target["node"]]
        if target["path"] == "rotation":
            local_rotations[:, joint] = values
        elif target["path"] == "translation" and joint == 0:
            root_translations[:] = values
    if common_times is None:
        raise ValueError("animation contains no supported skin channels")
    fps = float((frame_count - 1) / common_times[-1]) if frame_count > 1 and common_times[-1] > 0 else 30.0

    frames = selected_frames(args.frames, frame_count)
    homogeneous = np.concatenate((positions.astype(np.float64), np.ones((len(positions), 1))), axis=1)
    expected = np.zeros((len(frames), len(positions), 3), dtype="<f4")
    for output_index, frame in enumerate(frames):
        globals_ = np.zeros((joint_count, 4, 4), dtype=np.float64)
        for joint in range(joint_count):
            translation = root_translations[frame] if joint == 0 else node_translations[joint]
            local = local_matrix(translation, local_rotations[frame, joint])
            globals_[joint] = local if parents[joint] < 0 else globals_[parents[joint]] @ local
        skin_matrices = globals_ @ inverse_bind
        result = np.zeros((len(positions), 3), dtype=np.float64)
        weight_sum = weights.astype(np.float64).sum(axis=1)
        for slot in range(4):
            transformed = np.einsum("nij,nj->ni", skin_matrices[joints[:, slot]], homogeneous)[:, :3]
            result += transformed * weights[:, slot, None]
        result /= np.maximum(weight_sum[:, None], 1.0e-30)
        expected[output_index] = result.astype("<f4")

    args.output.mkdir(parents=True, exist_ok=True)
    positions.tofile(args.output / "positions.f32")
    joints.tofile(args.output / "joints.u16")
    weights.tofile(args.output / "weights.f32")
    faces.tofile(args.output / "faces.u32")
    parents.tofile(args.output / "parents.i32")
    rest_positions.tofile(args.output / "rest-positions.f32")
    root_translations.astype("<f4").tofile(args.output / "root-translations.f32")
    local_rotations.astype("<f4").tofile(args.output / "local-rotations.f32")
    expected.tofile(args.output / "expected-vertices.f32")
    manifest = {
        "format": 1,
        "reference": "independent NumPy glTF linear-blend skinning",
        "source_glb": args.glb.name,
        "source_sha256": hashlib.sha256(args.glb.read_bytes()).hexdigest(),
        "vertex_count": len(positions),
        "joint_count": joint_count,
        "face_count": len(faces),
        "frame_count": frame_count,
        "frames_per_second": fps,
        "selected_frames": frames,
        "max_abs_tolerance": 2.0e-5,
        "relative_l2_tolerance": 2.0e-6,
    }
    (args.output / "animation-vertices.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
