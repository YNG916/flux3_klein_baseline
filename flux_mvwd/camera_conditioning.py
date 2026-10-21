"""Optional query-camera cues derived solely from supplied calibration."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

MATRIX_TEXT = "matrix_text"
BEV_FRUSTUM_TEXT = "bev_frustum_text_v1"


def camera_summary(camera, target_hw):
    """Optical-axis orientation and border-ray FOV; no scene/depth queries."""
    T = np.asarray(camera["camera_to_local_world"], dtype=float)
    K = np.asarray(camera["K"], dtype=float)
    h, w = target_hw
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Camera cues require finite positive focal lengths")
    forward = T[:3, 2]
    horizontal = np.linalg.norm(forward[:2])
    heading = float(np.rad2deg(np.arctan2(forward[1], forward[0]))) if horizontal > 1e-8 else None
    def ray(u, v):
        r = np.linalg.solve(K, [u, v, 1.0])
        return r / np.linalg.norm(r)
    def angle(a, b):
        return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1, 1))))
    return {"robot_id": camera["robot_id"], "center_local_m": T[:3, 3].tolist(),
            "forward_local_unit": forward.tolist(), "heading_deg_from_positive_x_toward_positive_y": heading,
            "pitch_deg_above_horizontal": float(np.rad2deg(np.arctan2(forward[2], horizontal))),
            "horizontal_fov_deg": angle(ray(0.5, K[1, 2]), ray(w - 0.5, K[1, 2])),
            "vertical_fov_deg": angle(ray(K[0, 2], 0.5), ray(K[0, 2], h - 0.5)),
            "overlay_horizontal_range_m": 2.0,
            "overlay_meaning": "projection of optical-axis and border rays; not a visible/free-space mask"}


def draw_query_camera(image, camera, world_to_bev, target_hw):
    """Draw a fixed two-meter ray fan, without estimating any occlusion."""
    T = np.asarray(camera["camera_to_world"], dtype=float)
    K = np.asarray(camera["K"], dtype=float)
    projection = np.asarray(world_to_bev)
    h, w = target_hw
    center = T[:, 3]
    start = (projection @ center)[:2]
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    endpoints = []
    for u in (0.5, K[0, 2], w - 0.5):
        ray = T[:3, :3] @ np.linalg.solve(K, [u, K[1, 2], 1.0])
        horizontal = np.linalg.norm(ray[:2])
        if horizontal <= 1e-8:
            endpoints.append(None)
            continue
        point = center.copy()
        point[:3] += ray * (2.0 / horizontal)
        endpoints.append((projection @ point)[:2])
    if endpoints[0] is not None and endpoints[2] is not None:
        draw.polygon([tuple(start), tuple(endpoints[0]), tuple(endpoints[2])], fill=(255, 0, 255, 45))
    for index, end in enumerate(endpoints):
        if end is not None:
            draw.line([tuple(start), tuple(end)], fill=(255, 0, 255, 255), width=4 if index == 1 else 2)
    draw.ellipse((*tuple(start - 7), *tuple(start + 7)), fill=(255, 0, 255, 255), outline=(255, 255, 255, 255), width=2)
    if endpoints[1] is not None:
        end = endpoints[1]
        direction = end - start
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction /= norm
            normal = np.array([-direction[1], direction[0]])
            draw.polygon([tuple(end), tuple(end - direction * 9 + normal * 5), tuple(end - direction * 9 - normal * 5)], fill=(255, 0, 255, 255))
    draw.text((start[0] + 10, start[1] + 6), f"CAM {camera['robot_id']}", fill=(255, 0, 255, 255), stroke_width=1, stroke_fill=(255, 255, 255, 255))
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def apply_camera_cues(condition, variant=MATRIX_TEXT):
    """Idempotent: interventions redraw from the original robot-annotated BEV."""
    if variant == MATRIX_TEXT:
        return condition
    if variant != BEV_FRUSTUM_TEXT or len(condition["cameras"]) != 1:
        raise ValueError("Camera cue variant requires one independent camera")
    if "camera_cue_base_image" not in condition:
        prefix, separator, encoded = condition["prompt"].partition("Conditions:\n")
        if not separator or json.loads(encoded) != condition["payload"]:
            raise ValueError("Condition prompt does not match payload")
        condition["camera_cue_base_image"] = condition["images"][1].copy()
        condition["camera_cue_base_prompt_prefix"] = prefix
    condition["camera_cue_variant"] = variant
    camera = condition["cameras"][0]
    summary = camera_summary(camera, condition["payload"]["target_view_hw"])
    condition["payload"]["query_camera_summary"] = summary
    condition["images"][1] = draw_query_camera(condition["camera_cue_base_image"], camera,
        condition["world_to_bev"], condition["payload"]["target_view_hw"])
    position = ", ".join(f"{v:.3f}" for v in summary["center_local_m"])
    forward = ", ".join(f"{v:.3f}" for v in summary["forward_local_unit"])
    heading = summary["heading_deg_from_positive_x_toward_positive_y"]
    heading_text = "undefined (vertical optical axis)" if heading is None else f"{heading:.2f} degrees from +X toward +Y"
    brief = (f"QUERY CAMERA: robot {camera['robot_id']}; local center ({position}) meters; "
             f"forward unit vector ({forward}); horizontal heading {heading_text}; "
             f"pitch {summary['pitch_deg_above_horizontal']:.2f} degrees above horizontal; "
             f"horizontal FOV {summary['horizontal_fov_deg']:.2f} degrees and vertical FOV {summary['vertical_fov_deg']:.2f} degrees. "
             "The MAGENTA CAM marker in Image 2 is the query camera center; its central arrow shows the optical direction, "
             "and the translucent ray fan shows the horizontal optical border rays projected into BEV for a fixed 2-meter range. "
             "This is a calibration annotation, not a scene object, an occlusion map or a free-space guarantee. "
             "Generate the perspective seen from this marked camera; do not reproduce the BEV view or draw its annotations. ")
    condition["prompt"] = (condition["camera_cue_base_prompt_prefix"] + brief + "Conditions:\n" +
        json.dumps(condition["payload"], separators=(",", ":")))
    return condition
