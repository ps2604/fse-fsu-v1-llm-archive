# FSE Physics Constants - Centralized defaults to prevent drift
DEFAULT_PHYSICS_CONSTANTS = {
    "field_coupling_strength": 0.1,
    "diffusion_coefficient": 0.01,
    "evolution_damping": 0.95,
    "boundary_reflection": 0.8,
    "field_coherence_threshold": 0.1,
    "conservation_enforcement": 0.9,
    "symmetry_preservation": 0.8,
    "causality_enforcement": 1.0
}

DEFAULT_BOUNDARY_CONDITIONS = {
    "type": "mixed",
    "dirichlet_faces": [],
    "neumann_faces": [],
    "periodic_faces": ["x", "y"],
    "reflection_coefficient": 0.8
}

DEFAULT_EVOLUTION_RATES = {
    "language": 0.05,
    "vision": 0.03,
    "seismic": 0.08,
    "fluid": 0.04,
    "molecular": 0.02
}