import io
import os
import pickle
import platform
import sys
from typing import Dict, Any

import lmdb
import numpy as np
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from tqdm import tqdm


# --- Helper Functions for Data Compression ---


def _q(v: float, nd: int = 3) -> float:
    """Round float with fixed precision to save space."""
    return round(float(v), nd)


def _compress_object(o: Dict[str, Any], nd: int = 3) -> Dict[str, Any]:
    """
    Return a compact representation of an AI2-THOR object.
    Maintains the original key structure (e.g., axisAlignedBoundingBox)
    so the loader acts as a drop-in replacement.
    """
    # Essential fields
    out: Dict[str, Any] = {
        "objectId": o.get("objectId"),
        "objectType": o.get("objectType"),
        "visible": True,  # We only store visible objects in the loop
        "position": {"x": _q(o["position"]["x"], nd), "y": _q(o["position"]["y"], nd), "z": _q(o["position"]["z"], nd)},
        "distance": _q(o.get("distance", 0.0), nd),
    }

    # Maintain original AABB structure (just rounded)
    # This allows the existing graph builder to find the keys it expects.
    aabb = o.get("axisAlignedBoundingBox")
    if isinstance(aabb, dict):
        c = aabb.get("center", {})
        s = aabb.get("size", {})
        out["axisAlignedBoundingBox"] = {
            "center": {"x": _q(c.get("x", 0), nd), "y": _q(c.get("y", 0), nd), "z": _q(c.get("z", 0), nd)},
            "size": {"x": _q(s.get("x", 0), nd), "y": _q(s.get("y", 0), nd), "z": _q(s.get("z", 0), nd)},
        }

    return out


def encode_image_to_bytes(image_array: np.ndarray, fmt: str = "JPEG", quality: int = 80) -> bytes:
    """
    Encodes a numpy array to image bytes (JPEG or PNG) in memory.
    JPEG Quality 80 is a sweet spot: ~10-15KB size, very few artifacts.
    """
    if image_array is None:
        return None

    # Ensure uint8
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)

    img = Image.fromarray(image_array)

    with io.BytesIO() as bio:
        # optimize=True reduces size for both PNG and JPEG without quality loss
        img.save(bio, format=fmt, quality=quality, optimize=True)
        return bio.getvalue()


def encode_depth_to_bytes(depth_array: np.ndarray, max_depth_m: float = 50.0) -> bytes:
    """
    Quantizes float depth (meters) to uint16 PNG bytes.
    Preserves precision (~0.7mm) while allowing lossless PNG compression.
    """
    if depth_array is None:
        return None

    # Clip and quantize to 16-bit integer (0..65535)
    depth_clip = np.clip(depth_array, 0.0, max_depth_m)
    depth_u16 = np.round(depth_clip * (65535.0 / max_depth_m)).astype(np.uint16)

    img = Image.fromarray(depth_u16)
    with io.BytesIO() as bio:
        img.save(bio, format="PNG", optimize=True)
        return bio.getvalue()


def make_key(x, z, rot):
    """Generates a consistent byte key for LMDB storage."""
    return f"{x:.2f}|{z:.2f}|{rot}".encode("ascii")


# --- Main Generation Function ---


def generate_transition_tables(
    num_floorplans: int = 30,
    floorplans: list = None,
    transition_tables_path: str = "../data/transition_tables",
    grid_size: float = 0.1,
    additional_images: bool = True,
    use_lmdb: bool = True,
    jpg_quality: int = 85,
    render_width: int = 300,  # 300x300 for flexible downsampling
    render_height: int = 300,
):
    """
    Precomputes compressed transition tables for AI2-THOR scenes.
    Stores data in a format compatible with 'CachedEvent' loader.
    """
    os.makedirs(transition_tables_path, exist_ok=True)

    if floorplans is None:
        # Default: Kitchens (FloorPlan 1-30, skipping some known broken ones if any)
        # floorplans = [f"FloorPlan{i}" for i in list(range(1, 6)) + list(range(7, 8)) + list(range(9, 31))][:num_floorplans]
        floorplans = [f"FloorPlan{i}" for i in list(range(1, num_floorplans + 1))]

    visibility_distance = 50.0

    # Initialize Controller with specific resolution
    controller_args = dict(
        visibilityDistance=visibility_distance,
        gridSize=grid_size,
        renderDepthImage=additional_images,
        renderInstanceSegmentation=additional_images,
        width=render_width,
        height=render_height,
    )

    if platform.system() == "Linux":
        controller = Controller(platform=CloudRendering, **controller_args)
    else:
        controller = Controller(**controller_args)

    # Define discrete rotation steps
    rotations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]

    for scene in floorplans:
        print(f"\nGenerating compressed table for: {scene} | Res: {render_width}x{render_height}")
        controller.reset(scene=scene)

        reachable = controller.step("GetReachablePositions").metadata["actionReturn"]
        total_tasks = len(reachable) * len(rotations)

        if use_lmdb:
            output_path = os.path.join(transition_tables_path, f"{scene}.lmdb")
            # Map size: 1TB virtual address space (safe default)
            env = lmdb.open(output_path, map_size=1099511627776)
            txn = env.begin(write=True)
        else:
            output_path = os.path.join(transition_tables_path, f"{scene}.pkl")
            mapping = {}

        count = 0
        with tqdm(total=total_tasks, desc=f"Exploring {scene}") as pbar:
            for pos in reachable:
                x = round(pos["x"], 2)
                z = round(pos["z"], 2)

                for rot in rotations:

                    # Teleport agent to state
                    event = controller.step(
                        action="Teleport",
                        position={"x": pos["x"], "y": pos.get("y", 0), "z": pos["z"]},
                        rotation={"x": 0, "y": rot, "z": 0},
                        horizon=0,
                        standing=True,
                        forceAction=True,
                    )

                    # --- COMPRESSION & STRUCTURING ---

                    # 1. Structure Metadata
                    # We rebuild the metadata dict so it looks exactly like a real THOR event
                    # This avoids complex mapping logic in the loader.

                    # Filter only visible objects to save space
                    visible_objects = [_compress_object(obj) for obj in event.metadata["objects"] if obj["visible"]]

                    # Synthesize agent metadata (needed by SceneGraphBuilder)
                    agent_meta = {
                        "position": {"x": x, "y": 0.9, "z": z},  # y is approx
                        "rotation": {"x": 0, "y": rot, "z": 0},
                        "cameraHorizon": 0,
                    }

                    simulated_metadata = {
                        "objects": visible_objects,
                        "agent": agent_meta,
                        "lastActionSuccess": True,
                        "screenWidth": render_width,
                        "screenHeight": render_height,
                        # Pass grid size for compatibility checks later
                        "grid_size": grid_size,
                    }

                    # 2. Encode Images to Bytes
                    rgb_bytes = encode_image_to_bytes(event.frame, fmt="JPEG", quality=jpg_quality)

                    depth_bytes = None
                    seg_bytes = None

                    if additional_images:
                        depth_bytes = encode_depth_to_bytes(event.depth_frame, max_depth_m=visibility_distance)
                        seg_bytes = encode_image_to_bytes(event.instance_segmentation_frame, fmt="PNG")

                    # 3. Construct Final Payload
                    # 'metadata' is ready to use. '_bytes' fields need decoding on load.
                    state_data = {
                        "metadata": simulated_metadata,
                        "frame_bytes": rgb_bytes,
                        "depth_bytes": depth_bytes,
                        "seg_bytes": seg_bytes,
                    }

                    # --- STORAGE ---

                    if use_lmdb:
                        k_bytes = make_key(x, z, rot)
                        # We pickle the dictionary containing bytes and dicts
                        v_bytes = pickle.dumps(state_data)
                        txn.put(k_bytes, v_bytes)

                        count += 1
                        if count % 1000 == 0:
                            txn.commit()
                            txn = env.begin(write=True)
                    else:
                        mapping[(x, z, rot)] = state_data

                    pbar.update(1)

        # Final cleanup
        if use_lmdb:
            txn.commit()
            env.close()
            print(f"✅ Saved compressed table (LMDB): {output_path}")
        else:
            with open(output_path, "wb") as f:
                pickle.dump(mapping, f)
            print(f"✅ Saved compressed table (Pickle): {output_path}")

    controller.stop()


def set_working_directory():
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    current_directory = os.getcwd()
    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Working directory changed from '{current_directory}' to '{desired_directory}'")
        return
    print("Working directory:", os.getcwd())


if __name__ == "__main__":
    set_working_directory()
    # Example usage: Generate for one floorplan to test
    # generate_transition_tables(use_lmdb=True, grid_size=0.1)
    generate_transition_tables(use_lmdb=True, grid_size=0.1)
