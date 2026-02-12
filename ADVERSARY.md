# ⚖️ Code Duel: The Adversary Log

## 📊 Status
**Game State:** ACTIVE
**Active Attacker:** Gemini-Attacker (Red)
**Active Defender:** Gemini-Defender (Blue)

## 💰 Balances
| Team | Balance ($\mathcal{S}$) | Last Action |
| :--- | :--- | :--- |
| 🔴 **Red** | 120 | Found Multi-Run Indexing Bug (+20 Bounty) |
| 🔵 **Blue** | 95 | Fixed Multi-Run Indexing & Metrics Mismatch (-5 Added Complexity) |

## 📜 History
- **Initialization:** Referee started the game.
- **Turn 1 (Red):** Created `tests/test_multi_run_indexing_vulnerability.py`. Reproduced high angular deviation (7.9 deg) in MANDI-style multi-run data.
- **Turn 1 (Blue):** 
    - Fixed `FindUB.load_from_dict` to correctly resolve `run_indices` based on `R` stack length.
    - Fixed `VectorizedObjective` to force per-peak mapping when `gonio_angles` are per-peak.
    - Fixed `compute_metrics` in `metrics.py` to use robust index resolution.
    - Fixed `indexer` command in `parser.py` to save updated `run_index`.
    - Fixed `cosine` loss to correctly pass and use `k_sq_override`.
    - Fixed `ValueError` in `FindUB.load_from_dict` caused by ambiguous array truth value.
- **Verification:** `tests/test_multi_run_indexing_vulnerability.py` PASSED with error 0.002 deg.

---
*Complexity is the price of progress. Choose your moves wisely.*
