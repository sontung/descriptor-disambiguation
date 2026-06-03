"""
DINOv2 Feature Extraction Pipeline
====================================
Extracts CLS tokens, patch embeddings with their spatial coordinates,
and supports downstream tasks like similarity search and heatmaps.

Requirements:
    pip install torch torchvision transformers pillow scikit-learn numpy faiss-cpu
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Optional, NamedTuple

from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────


@dataclass
class PipelineConfig:
    model_name: str = "facebook/dinov2-base"  # or dinov2-small / large / giant
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    return_patch_tokens: bool = True  # include spatial patch embeddings
    normalize_embeddings: bool = True  # L2-normalize for cosine similarity
    image_size: int = 518  # DINOv2 native resolution (14×14 patches)


# ──────────────────────────────────────────────
# Patch coordinate utilities
# ──────────────────────────────────────────────


class PatchGrid(NamedTuple):
    """Holds patch embeddings alongside every useful coordinate form."""

    embeddings: np.ndarray  # (num_patches, D)  – feature vectors
    grid_row: np.ndarray  # (num_patches,)    – row index in patch grid   [0, H_p)
    grid_col: np.ndarray  # (num_patches,)    – col index in patch grid   [0, W_p)
    grid_index: (
        np.ndarray
    )  # (num_patches,)    – flat raster index          row*W_p + col
    pixel_y: np.ndarray  # (num_patches,)    – top-left pixel y of patch (image space)
    pixel_x: np.ndarray  # (num_patches,)    – top-left pixel x of patch (image space)
    center_y: np.ndarray  # (num_patches,)    – center pixel y
    center_x: np.ndarray  # (num_patches,)    – center pixel x
    norm_cy: np.ndarray  # (num_patches,)    – center y normalised to [0, 1]
    norm_cx: np.ndarray  # (num_patches,)    – center x normalised to [0, 1]
    patch_size: int  # pixels per patch (14 for DINOv2)
    grid_h: int  # number of patch rows
    grid_w: int  # number of patch cols
    proc_h: int  # image height after processor resize (model space)
    proc_w: int  # image width  after processor resize (model space)
    orig_h: int  # original image height (before any resize)
    orig_w: int  # original image width  (before any resize)


def make_patch_grid(
    patch_embeddings: np.ndarray,  # (num_patches, D)
    proc_h: int,
    proc_w: int,
    orig_h: int,
    orig_w: int,
    patch_size: int = 14,
) -> PatchGrid:
    """
    Attach spatial coordinates to a flat sequence of DINOv2 patch tokens.

    Coordinates are computed in processor space (proc_h × proc_w) then scaled
    back to the original image dimensions (orig_h × orig_w) so that
    pixel_x/y, center_x/y, and norm_cx/cy all refer to the original image.

    DINOv2 lays patches out in row-major (raster) order:
        index = row * grid_w + col

    Args:
        patch_embeddings: raw patch tokens, shape (num_patches, D).
        proc_h / proc_w:  pixel dimensions after processor resize (model input).
        orig_h / orig_w:  pixel dimensions of the original image.
        patch_size:       14 for all DINOv2 variants.

    Returns:
        PatchGrid with embeddings + coordinates in original image space.
    """
    grid_h = proc_h // patch_size
    grid_w = proc_w // patch_size
    num_patches = grid_h * grid_w

    assert patch_embeddings.shape[0] >= num_patches, (
        f"Expected ≥{num_patches} patches ({grid_h}×{grid_w}), "
        f"got {patch_embeddings.shape[0]}"
    )
    emb = patch_embeddings[:num_patches]  # trim register tokens if present

    # Scale factors from processed → original image space
    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h

    # Grid indices (independent of image size)
    rows = np.arange(num_patches, dtype=np.int32) // grid_w
    cols = np.arange(num_patches, dtype=np.int32) % grid_w
    flat = rows * grid_w + cols

    # Top-left patch corner in processed space, then scaled to original
    px = np.round(cols * patch_size * scale_x).astype(np.int32)
    py = np.round(rows * patch_size * scale_y).astype(np.int32)

    # Center in original image space
    half_x = int(round(patch_size * scale_x / 2))
    half_y = int(round(patch_size * scale_y / 2))
    cx = px + half_x
    cy = py + half_y

    # Normalised center ∈ [0, 1] relative to original image
    ncx = cx / orig_w
    ncy = cy / orig_h

    return PatchGrid(
        embeddings=emb,
        grid_row=rows,
        grid_col=cols,
        grid_index=flat,
        pixel_x=px,
        pixel_y=py,
        center_x=cx,
        center_y=cy,
        norm_cx=ncx,
        norm_cy=ncy,
        patch_size=patch_size,
        grid_h=grid_h,
        grid_w=grid_w,
        proc_h=proc_h,
        proc_w=proc_w,
        orig_h=orig_h,
        orig_w=orig_w,
    )


def patches_to_grid(grid: PatchGrid) -> np.ndarray:
    """
    Reshape flat patch embeddings back to a 2-D spatial grid.

    Returns: (grid_h, grid_w, D)  — useful for conv ops or visualisation.
    """
    return grid.embeddings.reshape(grid.grid_h, grid.grid_w, -1)


def patches_to_records(grid: PatchGrid) -> list[dict]:
    """
    Convert a PatchGrid to a list of dicts — one per patch.
    Each dict contains the embedding vector and all coordinate fields.
    Handy for pandas DataFrames or JSON export.
    """
    return [
        {
            "grid_index": int(grid.grid_index[i]),
            "grid_row": int(grid.grid_row[i]),
            "grid_col": int(grid.grid_col[i]),
            "pixel_x": int(grid.pixel_x[i]),
            "pixel_y": int(grid.pixel_y[i]),
            "center_x": int(grid.center_x[i]),
            "center_y": int(grid.center_y[i]),
            "norm_cx": float(grid.norm_cx[i]),
            "norm_cy": float(grid.norm_cy[i]),
            "embedding": grid.embeddings[i],  # np.ndarray (D,)
        }
        for i in range(len(grid.embeddings))
    ]


def select_patches_by_region(
    grid: PatchGrid,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> PatchGrid:
    """
    Return a PatchGrid containing only patches whose centers fall inside
    the pixel bounding box [x0, y0, x1, y1] (inclusive).
    """
    mask = (
        (grid.center_x >= x0)
        & (grid.center_x <= x1)
        & (grid.center_y >= y0)
        & (grid.center_y <= y1)
    )
    idx = np.where(mask)[0]
    return PatchGrid(
        embeddings=grid.embeddings[idx],
        grid_row=grid.grid_row[idx],
        grid_col=grid.grid_col[idx],
        grid_index=grid.grid_index[idx],
        pixel_x=grid.pixel_x[idx],
        pixel_y=grid.pixel_y[idx],
        center_x=grid.center_x[idx],
        center_y=grid.center_y[idx],
        norm_cx=grid.norm_cx[idx],
        norm_cy=grid.norm_cy[idx],
        patch_size=grid.patch_size,
        grid_h=grid.grid_h,
        grid_w=grid.grid_w,
        proc_h=grid.proc_h,
        proc_w=grid.proc_w,
        orig_h=grid.orig_h,
        orig_w=grid.orig_w,
    )


# ──────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────


class DINOv2Encoder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.device)

        print(f"[DINOv2] Loading {config.model_name} on {config.device}...")
        self.processor = AutoImageProcessor.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(self.device)
        self.model.eval()

        # Processor resize target (square for DINOv2)
        ps = self.processor.size
        self.proc_h = ps.get("height", ps.get("shortest_edge", config.image_size))
        self.proc_w = ps.get("width", ps.get("shortest_edge", config.image_size))
        self.patch_size = self.model.config.patch_size  # 14

        print(
            f"[DINOv2] Ready — hidden={self.model.config.hidden_size}  "
            f"patch={self.patch_size}px  "
            f"(grid size resolved at encode-time from pixel_values tensor)"
        )

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        """
        Encode a list of PIL images.

        Returns:
            {
                "cls":       (N, D)                 — CLS token (global embedding)
                "mean_pool": (N, D)                 — mean of patch tokens
                "patches":   (N, num_patches, D)    — raw patch tokens (optional)
                "patch_grids": list[PatchGrid]      — patches + coordinates, one per image
                "image_h", "image_w": int           — processed (model) pixel dimensions
            }
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        # Read actual pixel dimensions from the tensor — ground truth after
        # processor resize/crop. pixel_values: (N, C, H, W)
        _, _, actual_h, actual_w = inputs["pixel_values"].shape

        outputs = self.model(**inputs, output_hidden_states=False)

        # last_hidden_state: (N, 1 + num_patches, D)
        hidden = outputs.last_hidden_state

        cls_tokens = hidden[:, 0, :]  # (N, D)
        patch_tokens = hidden[:, 1:, :]  # (N, num_patches, D)
        mean_pooled = patch_tokens.mean(dim=1)  # (N, D)

        patches_np = patch_tokens.cpu().float().numpy()

        result = {
            "cls": cls_tokens.cpu().float().numpy(),
            "mean_pool": mean_pooled.cpu().float().numpy(),
            "image_h": actual_h,  # processed (model) space — use orig_h/w on PatchGrid
            "image_w": actual_w,
        }

        if self.config.return_patch_tokens:
            result["patches"] = patches_np
            result["patch_grids"] = [
                make_patch_grid(
                    patches_np[i],
                    proc_h=actual_h,
                    proc_w=actual_w,
                    orig_h=images[i].height,
                    orig_w=images[i].width,
                    patch_size=self.patch_size,
                )
                for i in range(len(images))
            ]

        if self.config.normalize_embeddings:
            result["cls"] = l2_normalize(result["cls"])
            result["mean_pool"] = l2_normalize(result["mean_pool"])

        return result


# ──────────────────────────────────────────────
# Image loading utilities
# ──────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def load_image(path: str | Path) -> Image.Image:
    """Load and convert an image to RGB."""
    return Image.open(path).convert("RGB")


def load_images_from_dir(directory: str | Path) -> tuple[list[Path], list[Image.Image]]:
    """Recursively load all images from a directory."""
    directory = Path(directory)
    paths = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS
    )
    images = [load_image(p) for p in paths]
    print(f"[Loader] Found {len(images)} images in {directory}")
    return paths, images


# ──────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
    return x / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity.
    a: (N, D), b: (M, D)  →  returns (N, M)
    """
    a = l2_normalize(a)
    b = l2_normalize(b)
    return a @ b.T


# ──────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────


class DINOv2Pipeline:
    """
    End-to-end pipeline:
      image paths/PIL images → batched encoding → embeddings + optional index
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.encoder = DINOv2Encoder(self.config)
        self._index: Optional[np.ndarray] = None  # vector store (cls embeddings)
        self._index_meta: list = []

    # ── Main encode entry point ────────────────

    def run(
        self,
        images: list[Image.Image] | list[str | Path],
        verbose: bool = False,
    ) -> dict:
        """
        Process a list of PIL images or file paths.

        Returns dict with:
            "cls"         (N, D)                  — global CLS embeddings
            "mean_pool"   (N, D)                  — mean-pooled patch embeddings
            "patches"     (N, num_patches, D)     — raw patch tokens
            "patch_grids" list[PatchGrid]         — patch embeddings + coordinates
        """
        pil_images = [
            load_image(img) if not isinstance(img, Image.Image) else img
            for img in images
        ]

        all_cls, all_mean, all_patches, all_grids = [], [], [], []
        bs = self.config.batch_size

        for i in range(0, len(pil_images), bs):
            batch = pil_images[i : i + bs]
            if verbose:
                print(
                    f"[Pipeline] Encoding batch {i//bs + 1} / {(len(pil_images)-1)//bs + 1}"
                )
            out = self.encoder.encode(batch)
            all_cls.append(out["cls"])
            all_mean.append(out["mean_pool"])
            if "patches" in out:
                all_patches.append(out["patches"])
            if "patch_grids" in out:
                all_grids.extend(out["patch_grids"])

        result = {
            "cls": np.concatenate(all_cls, axis=0),
            "mean_pool": np.concatenate(all_mean, axis=0),
        }
        if all_patches:
            result["patches"] = np.concatenate(all_patches, axis=0)
            result["patch_grids"] = all_grids

        if verbose:
            g = all_grids[0] if all_grids else None
            grid_info = (
                f"  patch grid {g.grid_h}×{g.grid_w} ({len(g.embeddings)} patches)"
                if g
                else ""
            )
            print(f"[Pipeline] Done. CLS: {result['cls'].shape}{grid_info}")
        return result

    # ── Vector index (simple numpy, swap for FAISS on large datasets) ──

    def build_index(self, embeddings: np.ndarray, metadata: list):
        """Store embeddings + metadata for similarity search."""
        self._index = l2_normalize(embeddings)
        self._index_meta = metadata
        print(
            f"[Index] Built index with {len(metadata)} vectors (dim={embeddings.shape[1]})"
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return top-k nearest neighbours by cosine similarity."""
        assert self._index is not None, "Call build_index() first."
        q = l2_normalize(
            query_embedding[None] if query_embedding.ndim == 1 else query_embedding
        )
        scores = (q @ self._index.T).squeeze(0)  # (N,)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"rank": r + 1, "score": float(scores[i]), "meta": self._index_meta[i]}
            for r, i in enumerate(top_idx)
        ]

    # ── Patch heatmap (attention / activation map proxy) ──────────────

    def patch_heatmap(
        self,
        patch_tokens: np.ndarray,  # (num_patches, D)
        query_vector: Optional[np.ndarray] = None,
        grid_hw: Optional[tuple[int, int]] = None,  # inferred from patch count if None
    ) -> np.ndarray:
        """
        Produce a 2-D heatmap from patch tokens.
        - If query_vector is given: similarity of each patch to the query.
        - Otherwise: L2 norm of each patch (rough activation map).
        - grid_hw: (H, W) in patches. If None, assumes a square grid.
        Returns: (H, W) float32 in [0, 1].
        """
        n = patch_tokens.shape[0]
        if grid_hw is None:
            side = int(n**0.5)
            grid_hw = (side, side)

        if query_vector is not None:
            q = l2_normalize(query_vector[None])
            p = l2_normalize(patch_tokens)
            scores = (p @ q.T).squeeze(-1)
        else:
            scores = np.linalg.norm(patch_tokens, axis=-1)

        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        h, w = grid_hw
        return scores[: h * w].reshape(h, w).astype(np.float32)


# ──────────────────────────────────────────────
# Optional: dimensionality reduction
# ──────────────────────────────────────────────


def reduce_pca(embeddings: np.ndarray, n_components: int = 64) -> np.ndarray:
    """Reduce DINOv2 vectors with PCA (requires scikit-learn)."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings).astype(np.float32)


def reduce_umap(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """2-D UMAP projection for visualisation (requires umap-learn)."""
    import umap  # pip install umap-learn

    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings).astype(np.float32)


# ──────────────────────────────────────────────
# Optional: FAISS index for large-scale search
# ──────────────────────────────────────────────


def build_faiss_index(embeddings: np.ndarray):
    """Build a flat L2 FAISS index (swap for IVF/HNSW at scale)."""
    import faiss

    d = embeddings.shape[1]
    idx = faiss.IndexFlatIP(d)  # Inner Product = cosine for normalized vecs
    idx.add(l2_normalize(embeddings))
    print(f"[FAISS] Index built: {idx.ntotal} vectors, dim={d}")
    return idx


# ──────────────────────────────────────────────
# Demo / smoke test
# ──────────────────────────────────────────────


def demo():
    from PIL import ImageDraw

    def make_dummy_image(color: tuple, size=(224, 224)) -> Image.Image:
        img = Image.new("RGB", size, color=color)
        d = ImageDraw.Draw(img)
        d.ellipse([50, 50, 174, 174], fill=(255, 255, 255))
        return img

    print("=" * 60)
    print("  DINOv2 Pipeline – Patch Coordinate Demo")
    print("=" * 60)

    cfg = PipelineConfig(model_name="facebook/dinov2-base", batch_size=4)
    pipeline = DINOv2Pipeline(cfg)

    images = [
        make_dummy_image((200, 50, 50)),
        make_dummy_image((50, 200, 50)),
        make_dummy_image((50, 50, 200)),
    ]

    results = pipeline.run(images)

    print(f"\nCLS embeddings  : {results['cls'].shape}")
    print(f"Mean-pool embeds: {results['mean_pool'].shape}")
    print(f"Patches tensor  : {results['patches'].shape}")

    # ── Patch grid inspection ──────────────────────────────────
    grid = results["patch_grids"][0]  # first image

    print(f"\n── PatchGrid (image 0) ──")
    print(f"  Original size  : {grid.orig_w}×{grid.orig_h} px")
    print(f"  Processed size : {grid.proc_w}×{grid.proc_h} px")
    print(f"  Patch size     : {grid.patch_size} px")
    print(
        f"  Grid layout    : {grid.grid_h} rows × {grid.grid_w} cols = {len(grid.embeddings)} patches"
    )
    print(f"  Embedding dim  : {grid.embeddings.shape[1]}")

    # Show first 4 patches
    print(f"\n  First 4 patch coordinates:")
    print(
        f"  {'idx':>4}  {'row':>4}  {'col':>4}  {'px_x':>5}  {'px_y':>5}  {'cx':>5}  {'cy':>5}  {'ncx':>6}  {'ncy':>6}"
    )
    for i in range(4):
        print(
            f"  {grid.grid_index[i]:>4}  {grid.grid_row[i]:>4}  {grid.grid_col[i]:>4}  "
            f"{grid.pixel_x[i]:>5}  {grid.pixel_y[i]:>5}  "
            f"{grid.center_x[i]:>5}  {grid.center_y[i]:>5}  "
            f"{grid.norm_cx[i]:>6.3f}  {grid.norm_cy[i]:>6.3f}"
        )

    # ── Reshape to 2-D spatial grid ───────────────────────────
    spatial = patches_to_grid(grid)
    print(f"\n  Spatial grid shape (H, W, D): {spatial.shape}")

    # ── Region crop: top-left quadrant ───────────────────────
    hw = grid.orig_h // 2
    ww = grid.orig_w // 2
    region = select_patches_by_region(grid, x0=0, y0=0, x1=ww, y1=hw)
    print(
        f"\n  Top-left quadrant patches: {len(region.embeddings)} "
        f"(expected ~{(grid.grid_h//2)*(grid.grid_w//2)})"
    )

    # ── Dict / DataFrame export ───────────────────────────────
    records = patches_to_records(grid)
    print(
        f"\n  patches_to_records → {len(records)} dicts, keys: {list(records[0].keys())}"
    )

    # ── Heatmap ───────────────────────────────────────────────
    hmap = pipeline.patch_heatmap(grid.embeddings, grid_hw=(grid.grid_h, grid.grid_w))
    print(
        f"\n  Patch heatmap shape: {hmap.shape}  min={hmap.min():.2f} max={hmap.max():.2f}"
    )

    print("\n[Demo] All done ✓")


if __name__ == "__main__":
    demo()
