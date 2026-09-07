import dataclasses
import os
import sys
import time
import traceback
import weakref
from pathlib import Path

import imgui
import lietorch
import torch
import moderngl
import moderngl_window as mglw
import numpy as np
from in3d.camera import Camera, ProjectionMatrix, lookat
from in3d.pose_utils import translation_matrix
from in3d.color import hex2rgba
from in3d.geometry import Axis
from in3d.viewport_window import ViewportWindow
from in3d.window import WindowEvents
from in3d.image import Image
from moderngl_window import resources
from moderngl_window.timers.clock import Timer

from mast3r_slam.frame import Mode
from mast3r_slam.geometry import get_pixel_coords
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.visualization_utils import (
    Frustums,
    Lines,
    depth2rgb,
    image_with_text,
)
from mast3r_slam.config import load_config, config, set_global_config


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _parse_window_size(default: tuple[int, int]) -> tuple[int, int]:
    value = os.environ.get("MAST3R_SLAM_VIZ_WINDOW_SIZE", "")
    if not value:
        return default
    try:
        width, height = value.lower().replace(",", "x").split("x", 1)
        return max(320, int(width)), max(240, int(height))
    except Exception:
        return default


def _pose_matrix_np(T_WC) -> np.ndarray:
    matrix = T_WC.matrix()
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    return np.asarray(matrix, dtype=np.float32).reshape(-1, 4, 4)[0]


@dataclasses.dataclass
class WindowMsg:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5


class Window(WindowEvents):
    title = "Functional-SLAM"
    window_size = (1920, 1080)

    def __init__(
        self,
        states,
        keyframes,
        main2viz,
        viz2main,
        functional_graph_viz_state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctx.gc_mode = "auto"
        remote_fast = _env_bool("MAST3R_SLAM_VIZ_REMOTE_FAST", False)
        # bit hacky, but detect whether user is using 4k monitor
        self.scale = 1.0
        if self.wnd.buffer_size[0] > 2560:
            self.set_font_scale(2.0)
            self.scale = 2
        self.clear = hex2rgba("#1E2326", alpha=1)
        resources.register_dir((Path(__file__).parent.parent / "resources").resolve())

        self.line_prog = self.load_program("programs/lines.glsl")
        self.surfelmap_prog = self.load_program("programs/surfelmap.glsl")
        self.trianglemap_prog = self.load_program("programs/trianglemap.glsl")
        self.pointmap_prog = self.surfelmap_prog

        width, height = self.wnd.size
        self.camera = Camera(
            ProjectionMatrix(width, height, 60, width // 2, height // 2, 0.05, 100),
            lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        )
        self.axis = Axis(self.line_prog, 0.1, 3 * self.scale)
        self.frustums = Frustums(self.line_prog)
        self.lines = Lines(self.line_prog)
        self.fg_lines = Lines(self.line_prog)

        self.viewport = ViewportWindow("Scene", self.camera)
        self.state = WindowMsg()
        self.keyframes = keyframes
        self.states = states

        self.show_all = _env_bool("MAST3R_SLAM_VIZ_SHOW_ALL", True)
        self.show_keyframe_edges = _env_bool("MAST3R_SLAM_VIZ_SHOW_EDGES", not remote_fast)
        self.culling = True
        self.follow_cam = _env_bool("MAST3R_SLAM_VIZ_FOLLOW_CAM", True)
        self.unfollow_on_interaction = _env_bool(
            "MAST3R_SLAM_VIZ_UNFOLLOW_ON_INTERACTION",
            remote_fast,
        )

        self.depth_bias = 0.001
        self.frustum_scale = 0.05

        self.dP_dz = None

        self.line_thickness = 3
        self.show_keyframe = False
        self.show_camera_trajectory = False
        self.show_functional_graph = _env_bool("MAST3R_SLAM_VIZ_SHOW_FG", True)
        self.show_fg_nodes = _env_bool("MAST3R_SLAM_VIZ_SHOW_FG_NODES", True)
        self.show_fg_edges = _env_bool("MAST3R_SLAM_VIZ_SHOW_FG_EDGES", True)
        self.fg_depth_test = _env_bool("MAST3R_SLAM_VIZ_FG_DEPTH_TEST", False)
        self.fg_node_size = _env_float("MAST3R_SLAM_VIZ_FG_NODE_SIZE", 0.035, minimum=0.0)
        self.fg_edge_thickness = _env_float("MAST3R_SLAM_VIZ_FG_EDGE_THICKNESS", 4.0, minimum=0.1)
        self.functional_graph_viz_state = functional_graph_viz_state
        self._fg_snapshot_version = -1
        self._fg_nodes = []
        self._fg_edges = []
        self._fg_summary = {}
        self.show_curr_pointmap = _env_bool("MAST3R_SLAM_VIZ_SHOW_CURRENT", True)
        self.hide_current_after_frame = _env_int(
            "MAST3R_SLAM_VIZ_HIDE_CURRENT_AFTER_FRAME",
            -1,
        )
        self.right_curr_panel = _env_bool("MAST3R_SLAM_VIZ_RIGHT_CURR_PANEL", False)
        self.right_curr_panel_width_frac = _env_float(
            "MAST3R_SLAM_VIZ_RIGHT_CURR_PANEL_WIDTH_FRAC",
            0.34,
            minimum=0.15,
        )
        if self.right_curr_panel:
            print(
                "[viz] right Curr panel enabled "
                f"(width_frac={self.right_curr_panel_width_frac:.2f})"
            )
        self.show_axis = True
        self.point_stride = _env_int(
            "MAST3R_SLAM_VIZ_POINT_STRIDE",
            2 if remote_fast else 1,
            minimum=1,
        )
        self.max_render_keyframes = _env_int(
            "MAST3R_SLAM_VIZ_MAX_KEYFRAMES",
            0,
            minimum=0,
        )
        self.max_dirty_uploads_per_frame = _env_int(
            "MAST3R_SLAM_VIZ_MAX_DIRTY_UPLOADS",
            2 if remote_fast else 1000000,
            minimum=1,
        )
        self.pending_dirty_idx = []
        self.pending_dirty_seen = set()

        self.textures = dict()
        self.mtime = self.pointmap_prog.extra["meta"].resolved_path.stat().st_mtime
        self.curr_img, self.kf_img = Image(), Image()
        self.curr_img_np, self.kf_img_np = None, None

        self.main2viz = main2viz
        self.viz2main = viz2main
        self._set_pointmap_radius(_env_float("MAST3R_SLAM_VIZ_RADIUS", 0.005, minimum=0.0))
        self._sync_follow_cam_overlays()

    def _set_pointmap_radius(self, radius: float) -> None:
        for program in (self.surfelmap_prog, self.pointmap_prog):
            if "radius" in program:
                program["radius"].value = float(radius)

    def _sync_follow_cam_overlays(self) -> None:
        show_context = not bool(self.follow_cam)
        self.show_keyframe = show_context
        self.show_camera_trajectory = show_context

    def resize(self, width: int, height: int):
        super().resize(width, height)
        self.camera.resize(max(1, int(width)), max(1, int(height)))

    def _update_functional_graph_snapshot(self) -> None:
        if self.functional_graph_viz_state is None:
            return
        try:
            version = int(self.functional_graph_viz_state.get("version", 0))
        except Exception:
            return
        if version == self._fg_snapshot_version:
            return
        try:
            snapshot = self.functional_graph_viz_state.get("snapshot", {}) or {}
        except Exception:
            return
        self._fg_snapshot_version = version
        self._fg_nodes = list(snapshot.get("nodes", []) or [])
        self._fg_edges = list(snapshot.get("edges", []) or [])
        self._fg_summary = dict(snapshot.get("summary", {}) or {})
        self._fg_kf_idx = int(snapshot.get("kf_idx", -1))

    def _render_functional_graph(self) -> None:
        if not self.show_functional_graph:
            return
        self._update_functional_graph_snapshot()
        if self.show_fg_edges:
            for edge in self._fg_edges:
                try:
                    src = np.asarray(edge["src_pos"], dtype=np.float32).reshape(1, 3)
                    dst = np.asarray(edge["dst_pos"], dtype=np.float32).reshape(1, 3)
                    color = edge.get("color_rgba", [1.0, 1.0, 1.0, 1.0])
                    thickness = float(self.fg_edge_thickness) * self.scale
                    self.fg_lines.add(src, dst, thickness=thickness, color=color)
                except Exception:
                    continue
        if self.show_fg_nodes:
            node_size = float(self.fg_node_size)
            if node_size <= 0.0:
                return
            offsets = np.array(
                [
                    [-node_size, 0.0, 0.0],
                    [0.0, -node_size, 0.0],
                    [0.0, 0.0, -node_size],
                ],
                dtype=np.float32,
            )
            for node in self._fg_nodes:
                try:
                    center = np.asarray(node["pos"], dtype=np.float32).reshape(1, 3)
                    starts = center + offsets
                    ends = center - offsets
                    color = node.get("color_rgba", [0.82, 0.82, 0.82, 1.0])
                    self.fg_lines.add(
                        starts,
                        ends,
                        thickness=max(2.0, float(self.fg_edge_thickness) * 0.8) * self.scale,
                        color=color,
                    )
                except Exception:
                    continue

    def render(self, t: float, frametime: float):
        self._sync_follow_cam_overlays()
        self.viewport.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        if self.culling:
            self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.clear(*self.clear)

        self.ctx.point_size = 2
        if self.show_axis:
            self.axis.render(self.camera)

        curr_frame = self.states.get_frame()
        h, w = curr_frame.img_shape.flatten()
        self.frustums.make_frustum(h, w)

        self.curr_img_np = curr_frame.uimg.numpy()
        self.curr_img.write(self.curr_img_np)

        cam_T_WC = as_SE3(curr_frame.T_WC).cpu()
        curr_T_WC = _pose_matrix_np(cam_T_WC)
        curr_cam_center = curr_T_WC[:3, 3]
        if self.follow_cam:
            T_WC = curr_T_WC @ translation_matrix(np.array([0, 0, -2], dtype=np.float32))
            self.camera.follow_cam(np.linalg.inv(T_WC))
        else:
            self.camera.unfollow_cam()
        self.frustums.add(
            cam_T_WC,
            scale=self.frustum_scale,
            color=[0, 1, 0, 1],
            thickness=self.line_thickness * self.scale,
        )

        with self.keyframes.lock:
            N_keyframes = len(self.keyframes)
            dirty_idx = self.keyframes.get_dirty_idx()
        for idx in dirty_idx:
            idx_int = int(idx)
            if idx_int not in self.pending_dirty_seen:
                self.pending_dirty_idx.append(idx_int)
                self.pending_dirty_seen.add(idx_int)

        dirty_batch = self.pending_dirty_idx[: self.max_dirty_uploads_per_frame]
        self.pending_dirty_idx = self.pending_dirty_idx[self.max_dirty_uploads_per_frame :]
        for idx_int in dirty_batch:
            self.pending_dirty_seen.discard(idx_int)

        for kf_idx in dirty_batch:
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()
            X = self.frame_X(keyframe)
            C = keyframe.get_average_conf().cpu().numpy().astype(np.float32)

            if keyframe.frame_id not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures[keyframe.frame_id] = ptex, ctex, itex
                ptex, ctex, itex = self.textures[keyframe.frame_id]
                itex.write(keyframe.uimg.numpy().astype(np.float32).tobytes())

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())

        render_start = 0
        if self.max_render_keyframes > 0:
            render_start = max(0, N_keyframes - self.max_render_keyframes)
        trajectory_points = []
        for kf_idx in range(N_keyframes):
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()
            keyframe_T_WC = as_SE3(keyframe.T_WC.cpu())
            trajectory_points.append(_pose_matrix_np(keyframe_T_WC)[:3, 3])
            if kf_idx == N_keyframes - 1:
                self.kf_img_np = keyframe.uimg.numpy()
                self.kf_img.write(self.kf_img_np)

            color = [1, 0, 0, 1]
            if self.show_keyframe:
                self.frustums.add(
                    keyframe_T_WC,
                    scale=self.frustum_scale,
                    color=color,
                    thickness=self.line_thickness * self.scale,
                )

            textures = self.textures.get(keyframe.frame_id)
            if textures is not None and self.show_all and kf_idx >= render_start:
                ptex, ctex, itex = textures
                self.render_pointmap(keyframe.T_WC.cpu(), w, h, ptex, ctex, itex)

        if self.show_camera_trajectory:
            if len(trajectory_points) >= 2:
                pts = np.stack(trajectory_points, axis=0)
                self.lines.add(
                    pts[:-1],
                    pts[1:],
                    thickness=2.0 * self.scale,
                    color=[0.1, 0.85, 1.0, 1.0],
                )
            if len(trajectory_points) >= 1:
                last_kf = trajectory_points[-1][None]
                self.lines.add(
                    last_kf,
                    curr_cam_center[None],
                    thickness=2.0 * self.scale,
                    color=[1.0, 0.85, 0.1, 1.0],
                )

        if self.show_keyframe_edges:
            with self.states.lock:
                ii = torch.tensor(self.states.edges_ii, dtype=torch.long)
                jj = torch.tensor(self.states.edges_jj, dtype=torch.long)
                if ii.numel() > 0 and jj.numel() > 0:
                    T_WCi = lietorch.Sim3(self.keyframes.T_WC[ii, 0])
                    T_WCj = lietorch.Sim3(self.keyframes.T_WC[jj, 0])
            if ii.numel() > 0 and jj.numel() > 0:
                t_WCi = T_WCi.matrix()[:, :3, 3].cpu().numpy()
                t_WCj = T_WCj.matrix()[:, :3, 3].cpu().numpy()
                self.lines.add(
                    t_WCi,
                    t_WCj,
                    thickness=self.line_thickness * self.scale,
                    color=[0, 1, 0, 1],
                )
        curr_frame_id = int(getattr(curr_frame, "frame_id", -1))
        current_pointmap_visible = bool(
            self.show_curr_pointmap
            and self.states.get_mode() != Mode.INIT
            and not (
                self.hide_current_after_frame >= 0
                and curr_frame_id >= int(self.hide_current_after_frame)
            )
        )
        if current_pointmap_visible:
            if config["use_calib"]:
                curr_frame.K = self.keyframes.get_intrinsics()
            h, w = curr_frame.img_shape.flatten()
            X = self.frame_X(curr_frame)
            C = curr_frame.C.cpu().numpy().astype(np.float32)
            if "curr" not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures["curr"] = ptex, ctex, itex
            ptex, ctex, itex = self.textures["curr"]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())
            itex.write(depth2rgb(X[..., -1], colormap="turbo"))
            self.render_pointmap(
                curr_frame.T_WC.cpu(),
                w,
                h,
                ptex,
                ctex,
                itex,
                use_img=True,
                depth_bias=self.depth_bias,
            )

        self._render_functional_graph()
        self.lines.render(self.camera)
        self.frustums.render(self.camera)
        if self.show_functional_graph:
            if not self.fg_depth_test:
                self.ctx.disable(moderngl.DEPTH_TEST)
            self.fg_lines.render(self.camera)
            if not self.fg_depth_test:
                self.ctx.enable(moderngl.DEPTH_TEST)
        self.render_ui()

    def render_ui(self):
        self.wnd.use()
        imgui.new_frame()

        io = imgui.get_io()
        # get window size and full screen
        window_size = io.display_size
        imgui.set_next_window_size(window_size[0], window_size[1])
        imgui.set_next_window_position(0, 0)
        self.viewport.render()

        imgui.set_next_window_size(
            window_size[0] / 4, 15 * window_size[1] / 16, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_position(
            32 * self.scale, 32 * self.scale, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_focus()
        imgui.begin("GUI", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        new_state = WindowMsg()
        _, new_state.is_paused = imgui.checkbox("pause", self.state.is_paused)

        imgui.spacing()
        _, new_state.C_conf_threshold = imgui.slider_float(
            "C_conf_threshold", self.state.C_conf_threshold, 0, 10
        )

        imgui.spacing()

        _, self.show_all = imgui.checkbox("show all", self.show_all)
        imgui.same_line()
        _, self.follow_cam = imgui.checkbox("follow cam", self.follow_cam)
        self._sync_follow_cam_overlays()

        imgui.spacing()
        shader_options = [
            "surfelmap.glsl",
            "trianglemap.glsl",
        ]
        current_shader = shader_options.index(
            self.pointmap_prog.extra["meta"].resolved_path.name
        )

        for i, shader in enumerate(shader_options):
            if imgui.radio_button(shader, current_shader == i):
                current_shader = i

        selected_shader = shader_options[current_shader]
        if selected_shader != self.pointmap_prog.extra["meta"].resolved_path.name:
            self.pointmap_prog = self.load_program(f"programs/{selected_shader}")

        imgui.spacing()

        _, self.show_keyframe_edges = imgui.checkbox(
            "show_keyframe_edges", self.show_keyframe_edges
        )
        imgui.spacing()

        _, self.pointmap_prog["show_normal"].value = imgui.checkbox(
            "show_normal", self.pointmap_prog["show_normal"].value
        )
        imgui.same_line()
        _, self.culling = imgui.checkbox("culling", self.culling)
        if "radius" in self.pointmap_prog:
            _, self.pointmap_prog["radius"].value = imgui.drag_float(
                "radius",
                self.pointmap_prog["radius"].value,
                0.0001,
                min_value=0.0,
                max_value=0.1,
            )
        if "slant_threshold" in self.pointmap_prog:
            _, self.pointmap_prog["slant_threshold"].value = imgui.drag_float(
                "slant_threshold",
                self.pointmap_prog["slant_threshold"].value,
                0.1,
                min_value=0.0,
                max_value=1.0,
            )
        imgui.text(
            "follow cam overlays: "
            f"keyframe={int(self.show_keyframe)} "
            f"trajectory={int(self.show_camera_trajectory)}"
        )
        _, self.show_functional_graph = imgui.checkbox(
            "show_functional_graph", self.show_functional_graph
        )
        if self.functional_graph_viz_state is not None:
            _, self.show_fg_nodes = imgui.checkbox("fg_nodes", self.show_fg_nodes)
            imgui.same_line()
            _, self.show_fg_edges = imgui.checkbox("fg_edges", self.show_fg_edges)
            _, self.fg_depth_test = imgui.checkbox("fg_depth_test", self.fg_depth_test)
            _, self.fg_node_size = imgui.drag_float(
                "fg_node_size",
                self.fg_node_size,
                0.001,
                min_value=0.0,
                max_value=0.2,
            )
            _, self.fg_edge_thickness = imgui.drag_float(
                "fg_edge_thickness",
                self.fg_edge_thickness,
                0.1,
                min_value=0.1,
                max_value=20.0,
            )
            imgui.text(
                "fg: "
                f"kf={getattr(self, '_fg_kf_idx', -1)} "
                f"nodes={int(self._fg_summary.get('n_nodes', 0))} "
                f"edges={int(self._fg_summary.get('n_edges', 0))} "
                f"simple={int(bool(self._fg_summary.get('simplified_colors', False)))} "
                f"collapse={int(bool(self._fg_summary.get('collapse_ocu', False)))} "
                f"hideC={int(bool(self._fg_summary.get('collapse_hide_c', False)))} "
                f"tmpR={int(bool(self._fg_summary.get('show_tentative_remote', False)))}"
            )
        _, self.show_curr_pointmap = imgui.checkbox(
            "show_curr_pointmap", self.show_curr_pointmap
        )
        _, self.point_stride = imgui.drag_int(
            "point_stride", int(self.point_stride), 1, min_value=1, max_value=16
        )
        _, self.max_render_keyframes = imgui.drag_int(
            "max_render_keyframes (0=all)",
            int(self.max_render_keyframes),
            1,
            min_value=0,
            max_value=512,
        )
        _, self.show_axis = imgui.checkbox("show_axis", self.show_axis)
        _, self.line_thickness = imgui.drag_float(
            "line_thickness", self.line_thickness, 0.1, 10, 0.5
        )

        _, self.frustum_scale = imgui.drag_float(
            "frustum_scale", self.frustum_scale, 0.001, 0, 0.1
        )

        imgui.spacing()
        imgui.text(f"window: {int(window_size[0])} x {int(window_size[1])}")
        for width, height in ((960, 540), (1280, 720), (1600, 900), (1920, 1080)):
            if imgui.button(f"{width}x{height}"):
                self.wnd.size = (width, height)
            imgui.same_line()
        imgui.new_line()

        imgui.spacing()

        gui_size = imgui.get_content_region_available()
        tex_w, tex_h = self.curr_img.texture.size
        width_scale = gui_size[0] / max(1, tex_w)
        scale = min(self.scale, width_scale)
        size = (tex_w * scale, tex_h * scale)
        image_with_text(self.curr_img, size, "curr", same_line=False)
        imgui.spacing()
        image_with_text(self.kf_img, size, "kf", same_line=False)

        imgui.end()

        if self.right_curr_panel:
            panel_w = max(240.0, float(window_size[0]) * float(self.right_curr_panel_width_frac))
            tex_w, tex_h = self.curr_img.texture.size
            image_w = max(1.0, panel_w - 24.0 * self.scale)
            image_h = image_w * float(tex_h) / max(1.0, float(tex_w))
            panel_h = min(
                max(120.0, float(window_size[1]) - 64.0 * self.scale),
                image_h + 50.0 * self.scale,
            )
            panel_x = max(0.0, float(window_size[0]) - panel_w - 18.0 * self.scale)
            panel_y = 32.0 * self.scale
            always = getattr(imgui, "ALWAYS", 1)
            imgui.set_next_window_size(panel_w, panel_h, always)
            imgui.set_next_window_position(panel_x, panel_y, always)
            imgui.set_next_window_focus()
            curr_flags = (
                getattr(imgui, "WINDOW_NO_COLLAPSE", 0)
                | getattr(imgui, "WINDOW_NO_SAVED_SETTINGS", 0)
            )
            imgui.begin("Curr", flags=curr_flags)
            curr_region = imgui.get_content_region_available()
            curr_scale = min(
                curr_region[0] / max(1, tex_w),
                curr_region[1] / max(1, tex_h),
            )
            curr_size = (tex_w * curr_scale, tex_h * curr_scale)
            image_with_text(self.curr_img, curr_size, "curr", same_line=False)
            imgui.end()

        if new_state != self.state:
            self.state = new_state
            self.send_msg()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def send_msg(self):
        self.viz2main.put(self.state)

    def _mark_camera_interaction(self):
        if self.unfollow_on_interaction:
            self.follow_cam = False
            self._sync_follow_cam_overlays()

    def mouse_drag_event(self, x, y, dx, dy):
        super().mouse_drag_event(x, y, dx, dy)
        self._mark_camera_interaction()

    def mouse_scroll_event(self, x_offset, y_offset):
        super().mouse_scroll_event(x_offset, y_offset)
        if abs(x_offset) > 0 or abs(y_offset) > 0:
            self._mark_camera_interaction()

    def render_pointmap(self, T_WC, w, h, ptex, ctex, itex, use_img=True, depth_bias=0):
        w, h = int(w), int(h)
        point_stride = max(1, int(self.point_stride))
        ptex.use(0)
        ctex.use(1)
        itex.use(2)
        model = _pose_matrix_np(T_WC).T

        vao = self.ctx.vertex_array(self.pointmap_prog, [], skip_errors=True)
        vao.program["m_camera"].write(self.camera.gl_matrix())
        vao.program["m_model"].write(model)
        vao.program["m_proj"].write(self.camera.proj_mat.gl_matrix())

        vao.program["pointmap"].value = 0
        vao.program["confs"].value = 1
        vao.program["img"].value = 2
        vao.program["width"].value = w
        vao.program["height"].value = h
        if "render_stride" in vao.program:
            vao.program["render_stride"].value = point_stride
        vao.program["conf_threshold"] = self.state.C_conf_threshold
        vao.program["use_img"] = use_img
        if "depth_bias" in self.pointmap_prog:
            vao.program["depth_bias"] = depth_bias
        render_w = (w + point_stride - 1) // point_stride
        render_h = (h + point_stride - 1) // point_stride
        vao.render(mode=moderngl.POINTS, vertices=render_w * render_h)
        vao.release()

    def frame_X(self, frame):
        if config["use_calib"]:
            Xs = frame.X_canon[None]
            if self.dP_dz is None:
                device = Xs.device
                dtype = Xs.dtype
                img_size = frame.img_shape.flatten()[:2]
                K = frame.K
                p = get_pixel_coords(
                    Xs.shape[0], img_size, device=device, dtype=dtype
                ).view(*Xs.shape[:-1], 2)
                tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
                tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
                self.dP_dz = torch.empty(
                    p.shape[:-1] + (3, 1), device=device, dtype=dtype
                )
                self.dP_dz[..., 0, 0] = tmp1
                self.dP_dz[..., 1, 0] = tmp2
                self.dP_dz[..., 2, 0] = 1.0
                self.dP_dz = self.dP_dz[..., 0].cpu().numpy().astype(np.float32)
            return (Xs[..., 2:3].cpu().numpy().astype(np.float32) * self.dP_dz)[0]

        return frame.X_canon.cpu().numpy().astype(np.float32)


def run_visualization(cfg, states, keyframes, main2viz, viz2main, functional_graph_viz_state=None) -> None:
    debug_viz = os.environ.get("MAST3R_SLAM_VIZ_DEBUG", "0") == "1"
    window = None
    window_config = None
    try:
        set_global_config(cfg)

        config_cls = Window
        backend = os.environ.get("MAST3R_SLAM_VIZ_BACKEND", "glfw")
        samples = int(os.environ.get("MAST3R_SLAM_VIZ_SAMPLES", "4"))
        window_size = _parse_window_size(config_cls.window_size)
        target_fps = _env_int("MAST3R_SLAM_VIZ_TARGET_FPS", 0, minimum=0)
        if debug_viz:
            print(
                f"[VIZ] starting pid={os.getpid()} DISPLAY={os.environ.get('DISPLAY')} "
                f"backend={backend} samples={samples} size={window_size} target_fps={target_fps}",
                flush=True,
            )
        window_cls = mglw.get_local_window_cls(backend)

        window = window_cls(
            title=config_cls.title,
            size=window_size,
            fullscreen=False,
            resizable=True,
            visible=True,
            gl_version=(3, 3),
            aspect_ratio=None,
            vsync=True,
            samples=samples,
            cursor=True,
            backend=backend,
        )
        if debug_viz:
            print("[VIZ] window created", flush=True)
        window.print_context_info()
        mglw.activate_context(window=window)
        window.ctx.gc_mode = "auto"
        timer = Timer()
        window_config = config_cls(
            states=states,
            keyframes=keyframes,
            main2viz=main2viz,
            viz2main=viz2main,
            functional_graph_viz_state=functional_graph_viz_state,
            ctx=window.ctx,
            wnd=window,
            timer=timer,
        )
        # Avoid the event assigning in the property setter for now
        # We want the even assigning to happen in WindowConfig.__init__
        # so users are free to assign them in their own __init__.
        window._config = weakref.ref(window_config)

        # Swap buffers once before staring the main loop.
        # This can trigged additional resize events reporting
        # a more accurate buffer size
        window.swap_buffers()
        window.set_default_viewport()

        timer.start()

        while not window.is_closing:
            current_time, delta = timer.next_frame()

            if window_config.clear_color is not None:
                window.clear(*window_config.clear_color)

            # Always bind the window framebuffer before calling render
            window.use()

            window.render(current_time, delta)
            if not window.is_closing:
                window.swap_buffers()
            if target_fps > 0:
                delay = max(0.0, (1.0 / float(target_fps)) - float(delta))
                if delay > 0:
                    time.sleep(delay)

        state = window_config.state
        window.destroy()
        state.is_terminated = True
        viz2main.put(state)
    except BaseException as exc:  # noqa: BLE001
        print(f"[VIZ] FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        try:
            if window is not None:
                window.destroy()
        except Exception:
            pass
        try:
            state = window_config.state if window_config is not None else WindowMsg()
            state.is_terminated = True
            viz2main.put(state)
        except Exception:
            pass
        raise
