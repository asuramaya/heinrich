"""3D Gaussian Splatting (3DGS / 2DGS) round-trip identity transfer pipeline.

The .splat artifact is a 3D ground reference for an arbitrary subject,
constructed from a set of multi-view photographs. Conceptually it sits
alongside heinrich's other model-forensics artifacts:

    .mri    — model state capture (per-token residuals, attention, MLP)
    .shrt   — residual displacement profile per token
    .lora   — weight-space character direction (in-prior identity)
    .splat  — 3D world ground reference (out-of-prior identity)

The fundamental insight (cf. session notes on null-conditional contrast):
LoRA averages a set of 2D photographs into invariant 2D statistics — losing
the 3D structure that produced them. 3DGS averages the same photographs
into the manifold that actually generated them — a 3D surface — and the
variance across views constrains the surface more tightly. Same data,
opposite outcome, because the manifold of aggregation is the right one.

Pipeline (subject-agnostic — no person/face priors):

    arbitrary set of images
        ↓ COLMAP (sparse SfM) — recover camera poses + sparse points
        ↓ gsplat optimization (2DGS or 3DGS) — densify + refine
    .splat artifact
        ↓ render at requested camera + light
    splat-render of the subject
        ↓ diffusion conditioning (img2img / depth-controlnet / IP-Adapter)
    final image with subject in arbitrary scene

Implementation choice: gsplat (UC Berkeley / Nerfstudio team) supports both
3DGS and 2DGS through one API. Default is 2DGS — Gaussians forced onto
surfaces, better for surface-decoration features (vitiligo, scars, tattoos).
3DGS available via --impl 3dgs flag for cases where volumetric features
(skin subsurface scattering, fabric softness, hair) dominate.
"""
from __future__ import annotations
