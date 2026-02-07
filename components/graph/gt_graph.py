import json
import os

from components.graph.global_graph import GlobalSceneGraph
from components.graph.scene_graph import Edge, Node


class GTGraph(GlobalSceneGraph):
    def __init__(self):
        super().__init__()
        # Mapping: viewpoint_key -> list of {object_id: visibility} dicts (optional, not always loaded)
        self.viewpoint_to_objects = {}

    def add_viewpoint(self, viewpoint, nodes):
        if viewpoint in self.viewpoint_to_objects:
            raise ValueError(f"Viewpoint {viewpoint} already exists in the graph.")
        objects = []
        for key, node in nodes.items():
            objects.append({key: node.visibility})
        self.viewpoint_to_objects[viewpoint] = objects

    @classmethod
    def load_from_file(cls, path: str, include_viewpoints: bool = False):
        with open(path, "r") as f:
            data = json.load(f)
        graph = cls()
        if include_viewpoints:
            graph.viewpoint_to_objects = data.get("viewpoint_to_objects", {})
        for node in data["nodes"]:
            graph.add_node(
                Node(
                    object_id=(node["object_id"].replace(",", ".") if "," in node["object_id"] else node["object_id"]),
                    name=node["name"],
                    position=tuple(node["position"]),
                    visibility=1,
                    properties=node["properties"],
                )
            )
        for edge in data["edges"]:
            graph.add_edge(
                Edge(
                    source=(edge["source"].replace(",", ".") if "," in edge["source"] else edge["source"]),
                    target=(edge["target"].replace(",", ".") if "," in edge["target"] else edge["target"]),
                    relation=edge["relation"],
                )
            )

        return graph

    def save_to_file(self, path: str):
        import tempfile
        import shutil

        # Atomic write: write to temp file first, then rename
        # This prevents corruption if the process is interrupted or iCloud syncs mid-write
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)

        # Create temp file in same directory (for atomic rename on same filesystem)
        with tempfile.NamedTemporaryFile(mode='w', dir=dir_name, delete=False, suffix='.tmp') as f:
            temp_path = f.name
            try:
                json.dump(
                    {
                        "nodes": [
                            {
                                "object_id": str(node.object_id),
                                "name": node.name,
                                "position": node.position,
                                "visibility": 1,
                                "properties": node.properties,
                            }
                            for node in self.nodes.values()
                        ],
                        "edges": [{"source": str(edge.source), "target": str(edge.target), "relation": edge.relation} for edge in self.edges],
                        "viewpoint_to_objects": self.viewpoint_to_objects,
                    },
                    f,
                    indent=2,
                )
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
            except Exception as e:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise e

        # Atomic rename (replaces old file)
        shutil.move(temp_path, path)
