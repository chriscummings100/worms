# Worms — a small, beautiful artificial ecosystem

> Original planning overview, retained as design context. Setup and current implementation status are in the [project README](../README.md). Statements about unimplemented code describe this planning snapshot.

Start with [PROJECT_PLAN.md](PROJECT_PLAN.md). It contains the staged build, GPU architecture, energy rules, learning design, experiments, and acceptance checks.

![Original visual reference supplied by the user](../assets/user-reference-original.png)

**Visual goal:** retain these simple ellipsoids, little limbs and eyes; add translucent materials, gentle internal shading, controlled glow and subtle motion effects. The complexity of the silhouettes should stay close to this reference.

**World:** a flat rectangle wrapping left ↔ right and top ↔ bottom. There are no walls or blocking corners; creatures sense and interact continuously across the joins.

- [Visual references and provenance](VISUALS.md)
- [Proposed baseline configuration](../baseline.proposed.toml)
- [GPU tick diagram](../assets/tick-pipeline.png)
- [Sensory history diagram](../assets/sensory-history.png)
- [Learning and inheritance diagram](../assets/learning-inheritance.png)

This folder is a planning package, not an implemented simulator. Configuration values are starting hypotheses; throughput, learning effectiveness and ecological stability have not been measured.

Prepared 24 October 2026 from the requested continuation of **Programming project ideas**, the supplied SVG and pixel image, and current primary technical references. The original **download.svg** is preserved unchanged. No packages were installed for the simulator and no existing project code was changed.
