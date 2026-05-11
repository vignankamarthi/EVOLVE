"""EVOLVE: stochastic LLM-driven evolutionary discovery framework.

AI4Pain 2026 is the first case study. The framework will outlive this project.

Three-level structure:
  Level 0: candidate programs (mutated every iteration)
  Level 1: this framework (FunSearch + GENITOR + scoring + constraints + breakdown + meta)
  Level 2: framework.introspect (mutates Level 1's own structure every M iterations)

See FRAMEWORK.md for full design spec.
"""
__version__ = "0.0.0-scaffold"
