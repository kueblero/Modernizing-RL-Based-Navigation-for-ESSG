import argparse
import json
import os
from pathlib import Path


def validate_gt_graphs(delete_corrupted=False, regenerate=False):
    """
    Validate all GT graph JSON files and identify corrupted ones.

    Args:
        delete_corrupted: If True, delete corrupted files
        regenerate: If True, automatically regenerate corrupted files

    Returns:
        True if all graphs are valid (or were fixed), False otherwise
    """
    gt_graphs_dir = Path(__file__).parent.parent / "data" / "gt_graphs"

    if not gt_graphs_dir.exists():
        print(f"GT graphs directory not found: {gt_graphs_dir}")
        return False

    json_files = sorted(gt_graphs_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {gt_graphs_dir}")
        return False

    print(f"Validating {len(json_files)} GT graph files...\n")

    corrupted = []
    valid = []

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Basic validation
            if "nodes" not in data or "edges" not in data:
                corrupted.append((json_file, "Missing required fields"))
                print(f"❌ {json_file.name}: Missing required fields")
            else:
                valid.append(json_file.name)
                print(f"✅ {json_file.name}: Valid ({len(data['nodes'])} nodes, {len(data['edges'])} edges)")

        except json.JSONDecodeError as e:
            corrupted.append((json_file, f"JSON decode error at line {e.lineno}, col {e.colno}: {e.msg}"))
            print(f"❌ {json_file.name}: JSON decode error at line {e.lineno}, col {e.colno}")

        except Exception as e:
            corrupted.append((json_file, str(e)))
            print(f"❌ {json_file.name}: {str(e)}")

    print(f"\n{'='*60}")
    print(f"Summary: {len(valid)} valid, {len(corrupted)} corrupted")
    print(f"{'='*60}")

    if corrupted:
        print(f"\nFound {len(corrupted)} corrupted files:")
        for json_file, error in corrupted:
            floorplan = json_file.stem.replace("_graph", "")
            print(f"  - {floorplan}: {error}")

        # Delete corrupted files if requested
        if delete_corrupted:
            print(f"\n[DELETE] Removing {len(corrupted)} corrupted files...")
            for json_file, _ in corrupted:
                try:
                    json_file.unlink()
                    print(f"  Deleted: {json_file.name}")
                except Exception as e:
                    print(f"  Failed to delete {json_file.name}: {e}")

        # Regenerate if requested
        if regenerate:
            print(f"\n[REGENERATE] Regenerating {len(corrupted)} graphs...")

            # Extract floorplan names
            floorplans = [json_file.stem.replace("_graph", "") for json_file, _ in corrupted]

            try:
                from components.scripts.generate_gt_graphs import generate_gt_scene_graphs
                generate_gt_scene_graphs(floorplans=floorplans)
                print(f"\n[SUCCESS] Regenerated {len(floorplans)} graphs")
                return True
            except Exception as e:
                print(f"\n[ERROR] Failed to regenerate graphs: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("\nTo regenerate corrupted files, run:")
            print(f"  python -m components.scripts.validate_gt_graphs --regenerate")
            return False
    else:
        print("\n[SUCCESS] All GT graphs are valid!")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate GT graph JSON files")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete corrupted files"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Automatically regenerate corrupted files (implies --delete)"
    )

    args = parser.parse_args()

    # If regenerating, we also need to delete first
    delete_corrupted = args.delete or args.regenerate

    success = validate_gt_graphs(delete_corrupted=delete_corrupted, regenerate=args.regenerate)

    # Exit with non-zero code if validation failed
    exit(0 if success else 1)