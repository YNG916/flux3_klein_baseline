"""Supervision is consumed here, never by the scene-condition forward path."""

import torch
from torch import nn
import torch.nn.functional as F
from .geometry import yaw_matrix, resize_intrinsics
from .rendering import render


def masked_mean(value, valid):
    valid = valid.to(value)
    clean = torch.where(valid.bool(), value, torch.zeros_like(value))
    return clean.sum() / valid.sum().clamp_min(1)


class CanonicalLoss(nn.Module):
    def __init__(self, cfg, teacher_channels=384):
        super().__init__()
        self.cfg = cfg
        self.semantic = nn.Linear(cfg.feature_dim, cfg.semantic_classes)
        self.rgb = nn.Linear(cfg.feature_dim, 3)
        # Frozen orthogonal distillation target prevents jointly collapsing auxiliary heads.
        generator = torch.Generator().manual_seed(121)
        q, _ = torch.linalg.qr(
            torch.randn(teacher_channels, cfg.feature_dim, generator=generator),
            mode="reduced",
        )
        self.register_buffer("teacher_projection", q)

    def layout(self, pred, target):
        from scipy.optimize import linear_sum_assignment

        device = pred.centers.device
        obs = target["objects"]
        N = len(obs)
        Q = len(pred.classes)
        if N > Q:
            raise ValueError(
                f"{N} supervised instances exceed {Q} queries; increase capacity, never truncate labels"
            )
        categories = torch.tensor(
            [x["category"] for x in obs], device=device, dtype=torch.long
        )
        centers = (
            torch.stack([torch.as_tensor(x["center"], device=device) for x in obs])
            if obs
            else pred.centers.new_empty(0, 3)
        )
        sizes = (
            torch.stack([torch.as_tensor(x["aabb_size"], device=device) for x in obs])
            if obs
            else centers
        )
        if N:
            masksize = pred.masks.shape[-2:]
            instances = F.interpolate(
                target["bev_instance"][None, None].float(), masksize, mode="nearest"
            )[0, 0].long()
            labels = torch.stack(
                [instances == int(x["public_instance_id"]) for x in obs]
            ).float()
            p = pred.masks.sigmoid().flatten(1)
            gt = labels.flatten(1)
            dice = 1 - (2 * p @ gt.T + 1) / (p.sum(1)[:, None] + gt.sum(1)[None] + 1)
            cost = (
                -pred.classes.softmax(-1)[:, categories]
                + torch.cdist(pred.centers.detach(), centers, p=1) / self.cfg.far
                + dice.detach()
            )
            a, b = linear_sum_assignment(cost.detach().cpu().numpy())
            a = torch.as_tensor(a, device=device)
            b = torch.as_tensor(b, device=device)
        else:
            a = b = torch.empty(0, device=device, dtype=torch.long)
        cls = torch.full(
            (Q,), self.cfg.semantic_classes, device=device, dtype=torch.long
        )
        cls[a] = categories[b]
        weights = torch.ones(self.cfg.semantic_classes + 1, device=device)
        weights[-1] = 0.05
        loss = {"object_class": F.cross_entropy(pred.classes, cls, weight=weights)}
        zero = pred.centers.sum() * 0
        if N:
            loss["object_center"] = F.smooth_l1_loss(pred.centers[a], centers[b])
            var = pred.center_log_variance[a]
            loss["geometry_uncertainty"] = (
                0.5 * ((pred.centers[a] - centers[b]).square() * (-var).exp() + var)
            ).mean()
            yaw = torch.tensor([x["yaw"] for x in obs], device=device)
            gt_yaw = torch.stack((yaw.sin(), yaw.cos()), -1)
            loss["object_yaw"] = (1 - (pred.yaw[a] * gt_yaw[b]).sum(-1)).mean()
            gt_rotation = torch.stack(
                [
                    torch.as_tensor(ob["rotation"], device=device)
                    if "rotation" in ob
                    else yaw_matrix(torch.tensor(ob["yaw"], device=device))
                    for ob in obs
                ]
            )
            loss["object_orientation"] = (
                (pred.rotations[a] - gt_rotation[b]).square().sum((-1, -2)).mean()
            )
            R = pred.rotations[a]
            aabb = torch.einsum("nij,nj->ni", R.abs(), pred.sizes[a])
            loss["object_size"] = F.smooth_l1_loss(aabb, sizes[b])
            visible = labels[b].flatten(1).any(-1)
            loss["bev_instance"] = (
                F.binary_cross_entropy_with_logits(
                    pred.masks[a][visible], labels[b][visible]
                )
                if visible.any()
                else zero
            )
        else:
            for k in [
                "object_center",
                "object_yaw",
                "object_orientation",
                "object_size",
                "bev_instance",
            ]:
                loss[k] = zero
        hw = pred.bev_height.shape
        sem = F.interpolate(
            target["bev_semantic"][None, None].float(), hw, mode="nearest"
        )[0, 0].long()
        height = F.interpolate(target["bev_height"][None, None], hw, mode="nearest")[
            0, 0
        ]
        valid = F.interpolate(
            target["bev_height_valid"][None, None].float(), hw, mode="nearest"
        )[0, 0].bool() & torch.isfinite(height)
        occ = F.interpolate(target["bev_occupancy"][None, None], hw, mode="nearest")[
            0, 0
        ]
        loss["bev_semantic"] = F.cross_entropy(
            pred.bev_semantic[None], sem[None], ignore_index=0
        )
        loss["bev_height"] = masked_mean(
            F.smooth_l1_loss(
                pred.bev_height, torch.nan_to_num(height), reduction="none"
            ),
            valid,
        )
        loss["bev_occupancy"] = F.binary_cross_entropy_with_logits(
            pred.bev_occupancy, occ
        )
        mapping = {int(obs[int(j)]["public_instance_id"]): int(i) for i, j in zip(a, b)}
        return loss, mapping

    def geometry(self, pred, target, state=None):
        world = pred.state.geometry if state is None else state.geometry
        depths = target["depth_linear"]
        K = target["geometry_K"]
        cameras = target["modality_c2w"]["depth_linear"]
        # Geometry supervision at quarter resolution using that modality's actual calibration.
        hw = (depths.shape[-2] // 4, depths.shape[-1] // 4)
        gt = F.interpolate(
            torch.nan_to_num(depths, nan=0, posinf=0)[:, None], hw, mode="nearest-exact"
        )
        valid = (
            F.interpolate(
                (
                    torch.isfinite(depths)
                    & (depths > self.cfg.near)
                    & (depths < self.cfg.far)
                ).float()[:, None],
                hw,
                mode="nearest-exact",
            )
            > 0.5
        )
        r = render(
            world,
            resize_intrinsics(K, depths.shape[-2:], hw),
            cameras,
            *hw,
            self.cfg.render_backend,
            self.cfg.near,
            self.cfg.far,
        )
        loss = {
            "depth": masked_mean(
                (r.depth.clamp_min(0.01).log() - gt.clamp_min(0.01).log()).abs(), valid
            ),
            "coverage": masked_mean(-r.alpha.clamp_min(1e-6).log(), valid),
        }
        # Free-space rays constrain only observed space before a valid GT hit. Unknown volume is never filled with negatives.
        valid_points = valid[:, 0].nonzero()
        if len(valid_points):
            valid_points = valid_points[:: max(1, len(valid_points) // 512)][:512]
            loss["free_space"] = self.free_space(
                world,
                gt,
                resize_intrinsics(K, depths.shape[-2:], hw),
                cameras,
                valid_points,
            )
        else:
            loss["free_space"] = world.means.sum() * 0
        semK = resize_intrinsics(K, target["semantic"].shape[-2:], hw)
        sr = render(
            world,
            semK,
            target["modality_c2w"]["semantic"],
            *hw,
            self.cfg.render_backend,
            self.cfg.near,
            self.cfg.far,
        )
        labels = F.interpolate(
            target["semantic"][:, None].float(), hw, mode="nearest-exact"
        )[:, 0].long()
        logits = self.semantic(sr.features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        loss["view_semantic"] = F.cross_entropy(logits, labels, ignore_index=0)
        from .geometry import unproject

        normal_terms = []
        for i in range(3):
            kk = resize_intrinsics(K[i], depths.shape[-2:], hw)
            a = unproject(r.depth[i, 0], kk, cameras[i])
            b = unproject(gt[i, 0], kk, cameras[i])
            na = torch.linalg.cross(
                a[1:, :-1] - a[:-1, :-1], a[:-1, 1:] - a[:-1, :-1], dim=-1
            )
            nb = torch.linalg.cross(
                b[1:, :-1] - b[:-1, :-1], b[:-1, 1:] - b[:-1, :-1], dim=-1
            )
            mask = valid[i, 0, :-1, :-1] & valid[i, 0, 1:, :-1] & valid[i, 0, :-1, 1:]
            normal_terms.append(
                masked_mean(1 - F.cosine_similarity(na, nb, dim=-1), mask)
            )
        loss["surface_normal"] = torch.stack(normal_terms).mean()
        rgbK = resize_intrinsics(target["rgb_K"], target["rgb"].shape[-2:], hw)
        rr = render(
            world,
            rgbK,
            target["rgb_c2w"],
            *hw,
            self.cfg.render_backend,
            self.cfg.near,
            self.cfg.far,
        )
        prediction = (
            self.rgb(rr.features.permute(0, 2, 3, 1)).sigmoid().permute(0, 3, 1, 2)
        )
        truth = F.interpolate(target["rgb"], hw, mode="area")
        loss["coarse_rgb"] = (prediction - truth).abs().mean()
        return loss

    def free_space(self, world, depth, K, c2w, points):
        from .geometry import pixel_rays, transform, quaternion_matrix

        losses = []
        R = quaternion_matrix(world.quats)
        for v in range(3):
            selected = points[points[:, 0] == v]
            if not len(selected):
                continue
            y, x = selected[:, 1], selected[:, 2]
            rays = pixel_rays(K[v], *depth.shape[-2:])[y, x]
            # Stratified free points before 95% of the observed surface, not behind it.
            factors = depth.new_tensor([0.25, 0.5, 0.75, 0.9])
            cam = (
                rays[:, None]
                * depth[v, 0, y, x][:, None, None]
                * factors[None, :, None]
            )
            xyz = transform(cam.reshape(-1, 3), c2w[v])
            chunks = []
            for start in range(0, len(xyz), 64):
                delta = xyz[start : start + 64, None] - world.means[None]
                local = torch.einsum("pni,nij->pnj", delta, R) / world.scales[
                    None
                ].clamp_min(0.001)
                density = (
                    torch.exp(-0.5 * local.square().sum(-1)) * world.opacity[None]
                ).sum(-1)
                chunks.append(density)
            losses.append(torch.cat(chunks).mean())
        return torch.stack(losses).mean() if losses else world.means.sum() * 0

    def appearance(self, aux, target, pred, mapping):
        features = aux["shared_features"]
        v, c, h, w = features.shape
        teacher = target["teacher"].float() @ self.teacher_projection
        teacher = F.normalize(teacher, dim=-1).transpose(1, 2).reshape(v, c, h, w)
        coverage = aux["alpha"][:, 0] > 0.05
        losses = {
            "appearance_distill": masked_mean(
                1 - F.cosine_similarity(features, teacher, dim=1), coverage
            )
        }
        # Compute surface correspondences using actual depth-camera geometry, then sample RGB-feature planes.
        # Small modality pose differences are registered explicitly rather than assuming co-location.
        source = F.normalize(aux["features"], dim=1)
        rgbK = target["rgb_K"]
        rgbT = target["rgb_c2w"]
        depth = target["depth_linear"]
        depthK = target["geometry_K"]
        depthT = target["modality_c2w"]["depth_linear"]
        from .geometry import unproject, project, sample_map

        surface = []
        instance = []
        instance_features = {}
        for i in range(3):
            valid = (
                torch.isfinite(depth[i])
                & (depth[i] > self.cfg.near)
                & (depth[i] < self.cfg.far)
            )
            ys, xs = valid.nonzero(as_tuple=True)
            stride = max(1, len(ys) // 1024)
            ys, xs = ys[::stride][:1024], xs[::stride][:1024]
            if not len(ys):
                continue
            safe = torch.where(valid, depth[i], torch.ones_like(depth[i]))
            xyz = unproject(safe, depthK[i], depthT[i])[ys, xs]
            grid, _, inside = project(
                xyz,
                resize_intrinsics(rgbK[i], target["rgb"].shape[-2:], (h, w)),
                rgbT[i],
                h,
                w,
                self.cfg.near,
                self.cfg.far,
            )
            first = sample_map(source[i : i + 1], grid[None])[0]
            for j in range(i + 1, 3):
                g, z, ok = project(
                    xyz,
                    depthK[j],
                    depthT[j],
                    *depth[j].shape,
                    self.cfg.near,
                    self.cfg.far,
                )
                validj = torch.isfinite(depth[j]) & (depth[j] > 0)
                other = sample_map(
                    torch.where(validj, depth[j], torch.zeros_like(depth[j]))[
                        None, None
                    ],
                    g[None],
                )[0, :, 0]
                support = sample_map(validj.float()[None, None], g[None])[0, :, 0]
                rg, _, rin = project(
                    xyz,
                    resize_intrinsics(rgbK[j], target["rgb"].shape[-2:], (h, w)),
                    rgbT[j],
                    h,
                    w,
                    self.cfg.near,
                    self.cfg.far,
                )
                ok = (
                    ok
                    & inside
                    & rin
                    & (support > 1 - 1e-6)
                    & ((z - other).abs() < 0.05 + 0.01 * z.abs())
                )
                second = sample_map(source[j : j + 1], rg[None])[0]
                if ok.any():
                    surface.append(
                        (1 - F.cosine_similarity(first[ok], second[ok], dim=-1)).mean()
                    )
            # Instance map uses its own pose: project depth points into its actual camera.
            ig, _, iv = project(
                xyz,
                depthK[i],
                target["modality_c2w"]["instance"][i],
                *target["instance"][i].shape,
                self.cfg.near,
                self.cfg.far,
            )
            ids = F.grid_sample(
                target["instance"][i : i + 1, None].float(),
                ig[None, None],
                mode="nearest",
                align_corners=False,
            )[0, 0, 0].long()
            for public, slot in mapping.items():
                mask = (ids == public) & iv & inside
                if mask.any():
                    instance_features.setdefault(public, []).append(first[mask].mean(0))
        losses["surface"] = (
            torch.stack(surface).mean() if surface else features.sum() * 0
        )
        ids = sorted(instance_features)
        if ids:
            prototypes = F.normalize(
                pred.object_features[
                    torch.tensor([mapping[k] for k in ids], device=features.device)
                ],
                dim=-1,
            )
            for label, public in enumerate(ids):
                for observation in instance_features[public]:
                    score = F.normalize(observation, dim=-1) @ prototypes.T / 0.1
                    instance.append(
                        F.cross_entropy(
                            score[None], torch.tensor([label], device=features.device)
                        )
                    )
        losses["instance_discrimination"] = (
            torch.stack(instance).mean() if instance else features.sum() * 0
        )
        return losses

    def total(self, losses):
        weights = {
            "flow": 1.0,
            "object_class": 0.2,
            "object_center": 0.2,
            "object_size": 0.1,
            "object_yaw": 0.025,
            "geometry_uncertainty": 0.025,
            "object_orientation": 0.05,
            "bev_instance": 0.1,
            "bev_semantic": 0.1,
            "bev_height": 0.2,
            "bev_occupancy": 0.1,
            "depth": 0.2,
            "coverage": 0.05,
            "free_space": 0.05,
            "surface_normal": 0.05,
            "coarse_rgb": 0.1,
            "view_semantic": 0.1,
            "appearance_distill": 0.1,
            "surface": 0.1,
            "instance_discrimination": 0.05,
            "mvs_depth": 0.1,
            "refined_depth": 0.2,
            "geometry_step": 0.01,
        }
        return sum(value * weights.get(name, 1.0) for name, value in losses.items())
