# scene_graph/global_scene_graph.py

from components.graph.scene_graph import SceneGraph


class GlobalSceneGraph(SceneGraph):
    def __init__(self):
        super().__init__()
        # Track which viewpoints have already contributed to each object's visibility
        # This prevents the "spin-in-place" loophole where rotating at same position
        # artificially inflates visibility scores
        self.viewpoint_history = {}  # obj_id → set of viewpoints

    def __getstate__(self):
        """
        Reduce multiprocessing IPC overhead.

        The policy/encoder only needs nodes/edges; `viewpoint_history` is used only inside
        the environment for reward shaping and should not be sent across processes.
        """
        state = dict(self.__dict__)
        state.pop("viewpoint_history", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.viewpoint_history = {}

    def add_local_sg(self, local_sg: SceneGraph, current_viewpoint=None, alpha=0.9):
        """
        Add local scene graph to global scene graph with viewpoint-based aggregation.

        Args:
            local_sg: Local scene graph from current observation
            current_viewpoint: Tuple (x, z) representing agent's grid position (no rotation)
            alpha: Aggregation weight (default 0.9)

        Note:
            Visibility is only updated if this viewpoint hasn't been used before for this object.
            This prevents agents from exploiting the soft visibility aggregation by spinning
            in place to artificially boost scores.
        """
        for obj_id, node in local_sg.nodes.items():
            local_vis = node.visibility

            if obj_id not in self.nodes:
                # First time seeing this object
                self.add_node(node)
                if current_viewpoint is not None:
                    self.viewpoint_history[obj_id] = {current_viewpoint}
            else:
                # Object seen before - check if viewpoint is new
                if current_viewpoint is None:
                    # Legacy behavior: always update (for backwards compatibility)
                    global_vis = self.nodes[obj_id].visibility
                    self.nodes[obj_id].visibility = 1 - (1 - global_vis) * (1 - alpha * local_vis)
                else:
                    # Check if this viewpoint was already used for this object
                    if obj_id not in self.viewpoint_history:
                        self.viewpoint_history[obj_id] = set()

                    if current_viewpoint not in self.viewpoint_history[obj_id]:
                        # New viewpoint - update visibility
                        global_vis = self.nodes[obj_id].visibility
                        self.nodes[obj_id].visibility = 1 - (1 - global_vis) * (1 - alpha * local_vis)
                        self.viewpoint_history[obj_id].add(current_viewpoint)
                    # Else: same viewpoint, no update (prevents loophole)

        for edge in local_sg.edges:
            if edge not in self.edges:
                self.add_edge(edge)

    def get_visited_viewpoints(self):
        """
        Returns a set of all viewpoints that have been visited so far.

        Returns:
            set: All unique viewpoints from viewpoint_history
        """
        visited = set()
        for obj_viewpoints in self.viewpoint_history.values():
            visited.update(obj_viewpoints)
        return visited
