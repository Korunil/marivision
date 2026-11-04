"""
D:/MARIVISION/H0_workbench/eval_tools/inspect_seaclips_train_data.py
Diagnostic probe to inspect the exact schema and structures of:
  1. seaclips_horizon_tracks_train_semantic_v2.json
  2. seaclips-train.json
"""

import json
from pathlib import Path
from collections import Counter

TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
TRAIN_TRACKS = Path("D:/MARIVISION/H0_workbench/horizon_tracks/seaclips_horizon_tracks_train_semantic_v2.json")


def inspect_horizon_tracks():
    print("=" * 80)
    print("1. INSPECTING: seaclips_horizon_tracks_train_semantic_v2.json")
    print("=" * 80)

    if not TRAIN_TRACKS.exists():
        print(f"[ERROR] File not found: {TRAIN_TRACKS}")
        return

    with open(TRAIN_TRACKS, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Top-level type: {type(data)}")
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Top-level keys ({len(keys)} total): {keys[:10]}")
        
        # Check if envelope exists
        sample_key = keys[0]
        sample_val = data[sample_key]
        print(f"\nSample entry under '{sample_key}':")
        print(f"  Type: {type(sample_val)}")
        
        if isinstance(sample_val, dict):
            subkeys = list(sample_val.keys())
            print(f"  Subkeys count: {len(subkeys)}")
            print(f"  Sample subkeys: {subkeys[:5]}")
            sample_sub_val = sample_val[subkeys[0]]
            print(f"  Sample sub-value: {sample_sub_val} (Type: {type(sample_sub_val)})")
            
            # Analyze all vector types and lengths across the file
            val_types = Counter()
            val_lens = Counter()
            sample_anomalies = []
            
            for k, subdict in data.items():
                if isinstance(subdict, dict):
                    for fk, fv in subdict.items():
                        val_types[type(fv).__name__] += 1
                        if isinstance(fv, (list, tuple)):
                            val_lens[len(fv)] += 1
                            if len(fv) != 3 and len(sample_anomalies) < 5:
                                sample_anomalies.append((k, fk, fv))
                        else:
                            if len(sample_anomalies) < 5:
                                sample_anomalies.append((k, fk, fv))
                                
            print("\nDistribution of horizon vector types:")
            for t_name, count in val_types.items():
                print(f"  {t_name}: {count:,}")
            print("\nDistribution of list lengths (for list/tuple types):")
            for length, count in sorted(val_lens.items()):
                print(f"  Length {length}: {count:,}")
                
            if sample_anomalies:
                print("\nSample anomalous horizon records (length != 3 or non-list):")
                for item in sample_anomalies:
                    print(f"  Video: {item[0]} | Frame: {item[1]} | Value: {item[2]}")

        elif isinstance(sample_val, list):
            print(f"  List length: {len(sample_val)}")
            print(f"  Sample elements: {sample_val[:3]}")


def inspect_seaclips_train():
    print("\n" + "=" * 80)
    print("2. INSPECTING: seaclips-train.json")
    print("=" * 80)

    if not TRAIN_JSON.exists():
        print(f"[ERROR] File not found: {TRAIN_JSON}")
        return

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Top-level keys: {list(data.keys())}")
    
    # Images
    images = data.get("images", [])
    print(f"Total images: {len(images):,}")
    if images:
        print("Sample image record:")
        print(json.dumps(images[0], indent=2))
        print("First 5 image file_name paths:")
        for img in images[:5]:
            print(f"  ID {img['id']}: {img.get('file_name')}")

    # Videos
    videos = data.get("videos", [])
    print(f"\nTotal videos: {len(videos):,}")
    if videos:
        print("Sample video record:")
        print(json.dumps(videos[0], indent=2))

    # Annotations
    annotations = data.get("annotations", [])
    print(f"\nTotal annotations: {len(annotations):,}")
    if annotations:
        print("Sample annotation record:")
        print(json.dumps(annotations[0], indent=2))


if __name__ == "__main__":
    inspect_horizon_tracks()
    inspect_seaclips_train()