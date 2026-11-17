from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class WorldConfig:
    feature_dim: int = 128
    hidden_dim: int = 256
    object_queries: int = 1536
    gaussians_per_object: int = 32
    background_side: int = 24
    background_levels: int = 4
    decoder_layers: int = 4
    semantic_classes: int = 253
    injection_blocks: tuple = (7, 15, 23)
    mvs_depth_bins: int = 32
    near: float = 0.1
    far: float = 15.0
    min_height: float = -4.0
    max_height: float = 12.0
    geometry_step_fraction: float = 0.05
    render_backend: str = "gsplat"
    unroll_steps: int = 2
    checkpoint_blocks: bool = True
    seed: int = 42

    def validate(self):
        if self.feature_dim % 8 or self.hidden_dim % 32 or self.feature_dim > 384:
            raise ValueError("Attention dimensions must be divisible by eight")
        if (
            min(
                self.object_queries,
                self.gaussians_per_object,
                self.mvs_depth_bins,
                self.unroll_steps,
            )
            < 1
        ):
            raise ValueError("Positive representation capacities required")
        if self.min_height >= self.max_height:
            raise ValueError("Invalid world height range")
        if not 0 < self.near < self.far or self.render_backend not in (
            "gsplat",
            "reference",
        ):
            raise ValueError("Invalid clipping range or renderer")
        if not self.injection_blocks or tuple(
            sorted(set(self.injection_blocks))
        ) != tuple(self.injection_blocks):
            raise ValueError("Injection blocks must be unique and ordered")
        return self

    def to_dict(self):
        return asdict(self)
