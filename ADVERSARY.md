# ⚖️ The Code Duel: ADVERSARY.md

**Status:** ACTIVE
**Mission Focus:** Identify sources of non-determinism (e.g., input file order).

## 📊 Scoreboard
| Team | Role | Balance ($\mathcal{S}$) |
| :--- | :--- | :--- |
| **Red** | Offense | 120 |
| **Blue** | Defense | 412 |

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

## 🛠️ Evidence
- `tests/test_merger_non_determinism.py`: Proven order dependency in HDF5 merging.
- `src/subhkl/export.py`: Sorting added to `BaseConcatenateMerger`.
- `src/subhkl/integration.py`: Deterministic assembly added to parallel loops.
