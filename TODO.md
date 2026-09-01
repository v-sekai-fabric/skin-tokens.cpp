# TODO

## SOMA30 motion retargeting follow-ups

- Validate the final animated vertices, including neutral-pose invariance,
  isolated-joint probes, GLB round trips, influence locality, and bounded mesh
  edge stretching.
- Add foot locking and optional contact-aware root correction after the
  rest-frame transfer is validated on a broader set of generated humanoids.
- Generalize structural recognition beyond the currently supported VROID-like
  humanoid core without falling back to nearest-joint guessing.

## Potential follow-up: constrained skeleton generation

- Investigate soft VROID humanoid class conditioning and hard topology masks for
  SOMA30- or VROID52-compatible generation. TokenRig should still generate the
  mesh-fitted joint positions and learned skin codes.
- Score constrained results with the original unconstrained model, reporting
  topology, coordinate, and skin-code likelihoods separately. Calibrate those
  measurements against geometric and animated-vertex quality checks rather than
  treating perplexity alone as an out-of-distribution detector.
