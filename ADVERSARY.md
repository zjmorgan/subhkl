# ⚔️ The Code Duel: ADVERSARY State

## 📊 Scoreboard
| Team | Balance ($\mathcal{S}$) | Role | Status |
| :--- | :--- | :--- | :--- |
| **🔴 Red Team** | 120 | Attacker | Active |
| **🔵 Blue Team** | 102 | Defender | Active |

## 📜 Mission Log
### Turn 1: Red Team (Offense)
- **Action:** Found failing test `test_cosine_indexer_accuracy`.
- **Test Complexity:** 2
- **Bounty:** +20 $\mathcal{S}$
- **Official Review:** `tests/test_cosine_indexer_regression.py`
- **Result:** **TOUCHDOWN!** Bug confirmed (Median Angular Error: 0.5553 > 0.3).

### Turn 2: Blue Team (Defense) - SUCCESS
- **Objective:** Fix high angular error in `cosine` indexer.
- **Strategy:** Pin $|Q|^2$ and $|h|^2$ kernel scaling to initial values in `VectorizedObjective` to remove lattice parameter bias in derivative-free optimization.
- **Official Review:** `tests/test_cosine_indexer_regression.py::test_cosine_indexer_accuracy` **PASSED**.
- **Refactoring Bonus:** +4 $\mathcal{S}$ (Added complexity: 4 for pinning logic, but improved robustness).

## 🗃️ Codebase Baseline
- `src/subhkl/io/parser.py`: 115 $\mathcal{S}$
- `src/subhkl/optimization.py`: 241 $\mathcal{S}$ (Updated)
- `tests/test_cosine_indexer_regression.py`: 2 $\mathcal{S}$
