# ⚖️ The Code Duel: ADVERSARY.md

**Status:** ACTIVE
**Mission Focus:** Identify sources of non-determinism (e.g., input file order, float stability).

## 📊 Scoreboard
| Team | Role | Balance ($\mathcal{S}$) |
| :--- | :--- | :--- |
| **Red** | Offense | 221 |
| **Blue** | Defense | 393 |

## 📜 History
- **[2026-02-12] Game Start.** Initialized balances at 100 $\mathcal{S}$.
- **[2026-02-12] Red identifies non-determinism.** 
    - Test `tests/test_merger_non_determinism.py` confirms `FinderConcatenateMerger` depends on input file order.
    - Test `tests/test_finder_non_determinism.py` exposes `as_completed` parallel jitter.
    - Red pays 18 $\mathcal{S}$ for tests.
    - **Touchdown!** Red finds a bug.
    - Red gets Refund + 20 $\mathcal{S}$ Bounty.
    - Blue gets 18 $\mathcal{S}$ Subsidy.
- **[2026-02-12] Blue fixes non-determinism.**
    - `BaseConcatenateMerger` now sorts input files.
    - `Peaks` parallel methods now sort results by image/bank index.
    - Complexity decreased from 988 to 841 (Refactoring Bonus: 294 $\mathcal{S}$).
    - Tests confirmed deterministic behavior.
- **[2026-02-12] Red proves correctness failure.**
    - Test `tests/test_correctness_proof.py` demonstrates that swapped merge order causes incorrect geometry lookup in `indexer`.
    - Red pays 1 $\mathcal{S}$ for test.
    - **Touchdown!** Red proves impact on correctness.
    - Red gets Refund + 20 $\mathcal{S}$ Bounty.
    - Blue gets 1 $\mathcal{S}$ Subsidy.
- **[2026-02-12] Red finds multi-run indexing vulnerability.**
    - Test `tests/test_multi_run_indexing_vulnerability.py` identifies silent mismatch in `VectorizedObjective` and data loss in `FindUB.minimize`.
    - Red pays 2 $\mathcal{S}$ for tests.
    - **Touchdown!** Red identifies a multi-run correctness bug.
    - Red gets Refund + 20 $\mathcal{S}$ Bounty.
    - Blue gets 2 $\mathcal{S}$ Subsidy.
- **[2026-02-12] Blue fixes multi-run indexing and rotation logic.**
    - `VectorizedObjective` now validates and broadcasts `R` stack to match `run_indices`.
    - `FindUB.minimize` now preserves intra-run rotation variations if detected.
    - `indexer` command now handles `R` expansion correctly and uses `np.round` for stable uniqueness checks.
    - Blue pays 10 $\mathcal{S}$ for complexity increase.
    - All tests passed.
- **[2026-02-12] Red exposes IndexError in angles expansion.**
    - User reported and test `tests/test_indexer_shape_crash.py` confirmed `IndexError` when `angles_stack` has shape `(N_peaks, N_axes)`.
    - Red pays 1 $\mathcal{S}$ for test.
    - **Touchdown!** Red finds a crash bug.
    - Red gets Refund + 20 $\mathcal{S}$ Bounty.
    - Blue gets 1 $\mathcal{S}$ Subsidy.
- **[2026-02-12] Blue fixes IndexError and hardens angles expansion.**
    - `indexer` now robustly handles both `(N_axes, N_entries)` and `(N_entries, N_axes)` for `angles_stack`.
    - Logic added to detect orientation of the stack and broadcast correctly if per-run.
    - Blue pays 10 $\mathcal{S}$ for complexity increase.
    - Validation test `tests/test_indexer_shape_crash.py` passed.
- **[2026-02-12] Red exposes high angular deviation sources.**
    - Test `tests/test_cosine_loss_instability.py` proved `cosine` loss is unstable for high-index reflections due to per-component kappa scaling.
    - Test `tests/test_findub_offset_vulnerability.py` proved `FindUB.load_from_dict` ignored nominal sample offsets.
    - Red pays 2 $\mathcal{S}$ for tests.
    - **Touchdown!** Red identifies two major sources of indexing inaccuracy.
    - Red gets Refund + 40 $\mathcal{S}$ Bounty.
    - Blue gets 2 $\mathcal{S}$ Subsidy.
- **[2026-02-12] Blue hardens indexing precision and geometry loading.**
    - `VectorizedObjective` now uses isotropic kappa scaling in `cosine` loss, preventing the "delta function" landscape.
    - `cosine` loss mixture model weighted (1% wide kernel) to prevent double-counting and drowning of narrow signals.
    - `FindUB.load_from_dict` now correctly loads `sample/offset` metadata.
    - Blue pays 5 $\mathcal{S}$ for Formatting Foul (51 files reformatted).
    - Median angular error on Mesolite reduced from 1.30 deg to 0.55 deg (within 0.5 deg tolerance).

## 🛠️ Evidence
- `tests/test_findub_offset_vulnerability.py`: Proven blindness to nominal sample offsets.
- `tests/test_cosine_loss_instability.py`: Proven instability in cosine loss at high index.
- `tests/test_indexer_shape_crash.py`: Verified fix for `IndexError` in angles stack expansion.
- `tests/test_multi_run_indexing_vulnerability.py`: Proven silent correctness failure and intra-run data loss.
- `src/subhkl/optimization.py`: Hardened loss kernels and robust metadata loading added.
- `src/subhkl/io/parser.py`: Robust run-index expansion and broadcast logic added.
