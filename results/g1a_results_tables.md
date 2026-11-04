# G1a results (G1a.0)

| Learner | Set | Boxes | Pooled bal. acc | Macro-F1 | Fold bal. acc |
|---|---|---|---|---|---|
| hgb | all_A | 65542 | 0.2360 | 0.2088 | 0.142, 0.427, 0.316, 0.247, 0.328 |
| hgb | all_B | 65542 | 0.2376 | 0.2195 | 0.259, 0.307, 0.380, 0.418, 0.262 |
| hgb | g1_A | 18494 | 0.2073 | 0.2205 | 0.304, 0.284, 0.197, 0.113, 0.201 |
| hgb | g1_B | 18494 | 0.2458 | 0.2597 | 0.415, 0.217, 0.241, 0.158, 0.370 |
| hgb | g1_C | 18494 | 0.2777 | 0.2879 | 0.419, 0.210, 0.215, 0.393, 0.488 |
| logreg | all_A | 65542 | 0.2319 | 0.2195 | 0.250, 0.255, 0.292, 0.309, 0.236 |
| logreg | all_B | 65542 | 0.2357 | 0.2265 | 0.313, 0.303, 0.292, 0.317, 0.226 |
| logreg | g1_A | 18494 | 0.3620 | 0.3268 | 0.576, 0.256, 0.301, 0.287, 0.504 |
| logreg | g1_B | 18494 | 0.3938 | 0.3560 | 0.620, 0.261, 0.422, 0.503, 0.353 |
| logreg | g1_C | 18494 | 0.3221 | 0.3061 | 0.633, 0.233, 0.398, 0.218, 0.479 |

| Learner | Comparison | Mean Δ | Folds > 0 | Pass |
|---|---|---|---|---|
| hgb | delta_all_B_minus_A | +0.0332 | 3/5 | False |
| hgb | delta_g1_B_minus_A | +0.0603 | 4/5 | True |
| hgb | delta_g1_C_minus_B | +0.0649 | 3/5 | False |
| logreg | delta_all_B_minus_A | +0.0219 | 4/5 | True |
| logreg | delta_g1_B_minus_A | +0.0473 | 4/5 | True |
| logreg | delta_g1_C_minus_B | -0.0400 | 2/5 | False |

Per-class recall (HGB, all_B): {"ANIMAL": 0.06621392190152801, "BOAT": 0.5529468136080499, "LEISURE_VEHICLE": 0.0, "MARINE_MARKER": 0.06768229786558756, "OBJECT": 0.4969409008070815, "SAILING_VESSEL": 0.4797559413958043, "SHIP": 0.0}

Verdict: {"chance_all": 0.14285714285714285, "R1_geometry_informative": false, "R2_position_adds_to_size": false, "R3_curated_horizon_adds": false}
