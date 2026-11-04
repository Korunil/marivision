# G4b results (G4B.0)

| model | off0 tiled macro | off0 tiled 5 classes w/o ANIMAL | off0 ANIMAL AP | off5 tiled macro | ANIMAL AP pooled 0+5 (tiled) | ANIMAL GT confident >= 0.3 | best ANIMAL score median / max | ANIMAL FP >= 0.05 |
|---|---|---|---|---|---|---|---|---|
| R0_e15 | 0.3459 | 0.4016 | 0.0673 | 0.3459 | 0.0792 | 6/52 | 0.000 / 0.886 | 971 |
| G4F_e15 | 0.5011 | 0.6013 | 0.0005 | 0.5157 | 0.0005 | 1/52 | 0.000 / 0.351 | 2897 |
| CTRL | 0.4912 | 0.5893 | 0.0004 | 0.5134 | 0.0005 | 2/52 | 0.000 / 0.444 | 2254 |
| NOTILE | 0.4781 | 0.5737 | 0.0005 | 0.4973 | 0.0005 | 1/52 | 0.000 / 0.457 | 2283 |
| RFS | 0.4916 | 0.5898 | 0.0004 | 0.5090 | 0.0005 | 3/52 | 0.000 / 0.544 | 2961 |

| class (off0 tiled) | R0_e15 | G4F_e15 | CTRL | NOTILE | RFS |
|---|---|---|---|---|---|
| ANIMAL | 0.0673 | 0.0005 | 0.0004 | 0.0005 | 0.0004 |
| BOAT | 0.3605 | 0.4347 | 0.4495 | 0.4052 | 0.4287 |
| MARINE_MARKER | 0.3766 | 0.5372 | 0.5494 | 0.5254 | 0.5101 |
| OBJECT | 0.2254 | 0.3610 | 0.3359 | 0.3187 | 0.3706 |
| SAILING_VESSEL | 0.5008 | 0.9277 | 0.9392 | 0.9308 | 0.9474 |
| SHIP | 0.5450 | 0.7458 | 0.6726 | 0.6882 | 0.6922 |

Rule: ANIMAL AP (tiled, offset 0 + 5 pooled) >= 0.05 AND >= CTRL + 0.03, AND offset-0 tiled macro of the 5 other classes >= CTRL - 0.01.

- NOTILE: {"animal_ap_pooled": 0.0005447233325682151, "ctrl_animal_ap_pooled": 0.0005135899429230386, "others_off0": 0.5736583421682413, "ctrl_others_off0": 0.5893314383335237, "animal_fixed": false, "others_guard": false, "PASS": false, "macro_off0_tiled": 0.47812724544269314}
- RFS: {"animal_ap_pooled": 0.0005284799984739486, "ctrl_animal_ap_pooled": 0.0005135899429230386, "others_off0": 0.5897994281608082, "ctrl_others_off0": 0.5893314383335237, "animal_fixed": false, "others_guard": true, "PASS": false, "macro_off0_tiled": 0.49156851864457135}

**WINNER: none -- no recipe change; do not start G5, diagnose further**
