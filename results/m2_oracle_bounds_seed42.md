# ORACLE headroom (upper bounds, uses VAL GT - NOT results) - seed42

| variant | as is | O1 no background FP | O2 + no part FP | O3 + no wrong-class FP | O4 O3 with perfect relabel |
|---|---|---|---|---|---|
| F_plain | 0.3657 | 0.3771 | 0.3787 | 0.454 | 0.4586 |
| F_tiled | 0.3524 | 0.3736 | 0.3755 | 0.522 | 0.5164 |
| T_plain | 0.314 | 0.3288 | 0.3362 | 0.4633 | 0.5111 |
| T_tiled | 0.3797 | 0.402 | 0.4142 | 0.5829 | 0.5946 |
| R0e04_plain | 0.4241 | 0.47 | 0.4766 | 0.5484 | 0.5732 |
| R0e04_tiled | 0.4344 | 0.4824 | 0.4873 | 0.6232 | 0.6343 |

Upper bounds only: a real verifier recovers some fraction of the O1-O4 headroom, never more.
