# ECEPP evaluator

This package implements the classic ECEPP conformational-energy form in JAX,
with the ECEPP/3 atom-type and torsion parameter tables:

\[
E = \sum_{ij}\frac{332.06371 q_iq_j}{\epsilon r_{ij}}
  + \sum_{ij}\left(\frac{A_{ij}}{r_{ij}^{12}}-\frac{B_{ij}}{r_{ij}^{6}}\right)
  + \sum_{ij}\left(\frac{C_{ij}}{r_{ij}^{12}}-\frac{D_{ij}}{r_{ij}^{10}}\right)
  + \sum_l U_l\left[1+\cos(n_l\phi_l-\delta_l)\right].
\]

Distances are in Angstrom and energies in kcal/mol. As in classic ECEPP,
bond lengths and bond angles are fixed; there are no bond-stretch or angle-bend
terms. `ECEPPParameters` is an evaluator-ready representation: residue-template
typing and conversion of a particular ECEPP/2 or ECEPP/3 parameter distribution
belong in a separate compiler/loader.

`ECEPP3ParameterSet.compile()` applies the ECEPP/3 Slater-Kirkwood mixing rule,
hydrogen-bond type table, torsion classes, and the special half-strength 1-4
repulsive coefficient. Pair lists are explicit: pass ordinary nonbonded pairs
and 1-4 pairs separately, omitting 1-2 and 1-3 interactions.

`ECEPP3Library.from_smmp(path)` reads the SMMP `lib.sh3` residue list. SMMP's
license adds non-commercial/non-military restrictions to GPL-2.0, so that file
is deliberately loaded as external user-supplied data instead of being copied
into this MIT-licensed repository. See `SOURCES.md` for provenance and hashes.

`ECEPPPotential.energy_dict()` returns each term and `etotal`; the regular call
returns only `etotal`. Both single `(atoms, 3)` and batched `(batch, atoms, 3)`
coordinates are accepted.

`EyringInternalCoordinates` implements the fixed-bond ECEPP coordinate tree,
including SMMP's same-parent sibling propagation rule, and remains fully JAX
differentiable. The ACE-ALA10-NME gold alignment plus MC and overdamped
Langevin baselines live in `examples/case4_ecepp3_ala10`.

`ECEPPPotential.energy_and_force()` uses a cached fused value-and-gradient
executable and returns both outputs in one reverse pass. `force(topology)` is
overridden to use this path instead of reconstructing a gradient transform on
every call.

The regular total-energy path coalesces electrostatic, 12-6, and 12-10 terms by
atom pair and computes every pair distance once. The component path remains as
an independently testable reference. Run `scripts/benchmark_ecepp.py` to compare
forward and reverse-mode timings on the current JAX device.
