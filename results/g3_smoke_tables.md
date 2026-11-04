# G3 backbone smoke (G3.0.1)

Environment: {"python": "3.11.13", "torch": "2.5.1+cu121", "torchvision": "0.20.1+cu121", "timm": "1.0.26", "cuda": true, "gpu": "NVIDIA GeForce RTX 3060", "msda_backend": "pytorch"}

| Backbone | params M (bb) | fp32 640 s/it | fp32 640 GB | AMP 640 s/it | AMP 640 GB | AMP 1024 s/it | AMP 1024 GB | overfit drop | 15 ep AMP 640 h | 15 ep AMP 1024 h | × R50 | feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R50 (R0 control) | 40 (23) | 0.461 | 4.8 | 0.405 | 4.7 | 0.850 | 10.8 | 0.60 | 17.2 | 36.0 | 0.88 | yes |
| ConvNeXt-T in22k | 41 (28) | 0.500 | 5.4 | 0.383 | 4.7 | 0.919 | 10.8 | 0.59 | 16.2 | 38.9 | 0.83 | yes |
| ConvNeXt-S in22k | 63 (49) | 0.623 | 6.7 | 0.425 | 5.6 | 2.123 | 12.6 | 0.61 | 18.0 | 89.8 | 0.92 | yes |
| ConvNeXt-B in22k | 101 (88) | 0.763 | 8.0 | 0.462 | 6.6 | 11.983 | 14.1 | 0.60 | 19.6 | 507.0 | 1.00 | yes |
| Swin-T in22k | 41 (28) | 0.483 | 5.5 | 0.394 | 4.8 | 1.123 | 11.1 | 0.57 | 16.7 | 47.5 | 0.86 | yes |
| Swin-B in22k | 101 (87) | 0.728 | 8.1 | 0.508 | 6.9 | 4.039 | 15.0 | 0.60 | 21.5 | 170.9 | 1.10 | yes |

R50 fp32 15-epoch projection: 19.486099461162667

Rule: Feasible = finite loss and gradients, overfit loss drop >= 30%, and peak memory <= 11.0 GB at 640x640 batch 2 (fp32 or AMP). Among feasible candidates, the recommendation is the highest-capacity backbone whose projected 15-epoch AMP time at 640 is <= 4x the R50 fp32 time; the Director sets the GPU budget, so the full table is reported either way.

**Recommended: ConvNeXt-B in22k**

Projections are GPU compute only (data loading excluded); a 15-epoch run is ~10,155 iterations per epoch.
