import os
import platform

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from tqdm import tqdm

from components.graph.gt_graph import GTGraph
from components.graph.local_graph_builder import LocalSceneGraphBuilder


def generate_gt_scene_graphs(num_floorplans: int = 30, floorplans: list = None):
    save_dir = os.path.join(os.path.dirname(__file__), "..", "data", "gt_graphs")
    os.makedirs(save_dir, exist_ok=True)
    if floorplans is None:
        floorplans = [f"FloorPlan{i}" for i in list(range(1, num_floorplans + 1))]

    controller = Controller(platform=CloudRendering if platform.system() == "Linux" else None, visibilityDistance=50.0, gridSize=0.1)
    builder = LocalSceneGraphBuilder()

    rotations = [0, 45, 90, 135, 180, 225, 270, 315]

    for scene in floorplans:
        print(f"\nCreating GT scene graph for: {scene}")
        controller.reset(scene=scene)

        # Get reachable positions
        reachable_positions = controller.step("GetReachablePositions").metadata["actionReturn"]

        total_steps = len(reachable_positions) * len(rotations)

        with tqdm(total=total_steps, desc=f"Exploring {scene}") as pbar:
            gt_graph = GTGraph()
            for pos in reachable_positions:
                for rot in rotations:
                    controller.step(
                        action="Teleport", position=pos, rotation={"x": 0, "y": rot, "z": 0}, horizon=0, standing=True, forceAction=True
                    )

                    vp = {"position": {"x": round(pos["x"], 2), "z": round(pos["z"], 2)}, "rotation": rot}
                    viewpoint = f"{vp['position']}_{vp['rotation']}"

                    event = controller.step("Pass")
                    local_sg = builder.build_from_metadata(event.metadata)
                    gt_graph.add_local_sg(local_sg)
                    gt_graph.add_viewpoint(viewpoint, local_sg.nodes)

                    pbar.update(1)

        # Save graph to file
        output_path = os.path.join(save_dir, f"{scene}.json")
        gt_graph.save_to_file(output_path)

        print(f"✅ Saved: {output_path}")

    controller.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ground truth scene graphs for AI2THOR floorplans")
    parser.add_argument(
        "--floorplans", nargs="+", default=None, help="Specific floorplans to generate (e.g., FloorPlan1 FloorPlan2)"
    )
    parser.add_argument("--num", type=int, default=30, help="Number of floorplans to generate if --floorplans not specified")

    args = parser.parse_args()

    generate_gt_scene_graphs(num_floorplans=args.num, floorplans=args.floorplans)
