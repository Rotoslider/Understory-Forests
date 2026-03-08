# Scan Alignment Tool

Compare tree measurements from two scans of the same plot — even when the scans are in different coordinate systems.

## The Problem

SLAM-based scanners (Livox, Ouster, etc.) define their coordinate frame based on the scanner's starting position and orientation. If you scan the same plot twice — or with two different scanners — the resulting point clouds will be in different coordinate systems. The X and Y axes may point in completely different directions depending on which way the scanner was facing when the scan started.

You can align the clouds visually in software like CloudCompare, but if the alignment transform doesn't get saved into the exported file coordinates, each scan's stem map will appear rotated differently when processed in Understory.

The `align_scans.py` tool solves this by automatically finding the rotation and translation between two sets of tree positions and generating an overlay comparison.

## How It Works

1. Loads `tree_data.csv` from two Understory pipeline runs
2. Extracts tree base positions (x_tree_base, y_tree_base) from each scan
3. Tests all possible rotation angles (1-degree steps, then refines to 0.1 degrees) to find the angle that maximizes the number of matched tree pairs
4. Estimates a translation offset from the matched pairs
5. Generates:
   - **Scan_Alignment_Overlay.png** — Stem map with both scans overlaid, matched pairs connected by lines
   - **scan_match_report.csv** — Matched tree pairs with DBH, height, and position offset from both scans

## Usage

### Basic

Point it at two pipeline output directories:

```bash
source venv/bin/activate
python scripts/align_scans.py <output_dir_A> <output_dir_B>
```

Scan A is the reference — scan B gets rotated to match it. Output files are saved to scan A's directory by default.

### With Options

```bash
python scripts/align_scans.py <output_dir_A> <output_dir_B> \
    --match-radius 2.5 \
    --output-dir /path/to/save/results
```

You can also pass `tree_data.csv` file paths directly instead of directories:

```bash
python scripts/align_scans.py /path/to/run1/tree_data.csv /path/to/run2/tree_data.csv
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--match-radius` | 2.0 | Maximum distance (meters) to consider two trees a match |
| `--output-dir` | scan A directory | Where to save the overlay image and match report |

## Output

### Console Output

```
Scan A: southfork_livox — 145 trees
Scan B: southfork_ouster — 130 trees
Match radius: 2.0 m

Matches with no rotation: 41 / 145
Searching for best rotation angle...
Best rotation: 267.0°
Translation: dx=-0.13 m, dy=-0.31 m
Matched trees: 107 / 145 (scan A) and 107 / 130 (scan B)
Match distance — mean: 0.30 m, max: 1.94 m

=== DBH Comparison (matched trees) ===
  Mean DBH difference (B - A): -0.0164 m
  Std DBH difference:          0.1167 m
  Mean absolute difference:    0.0695 m
```

### Scan_Alignment_Overlay.png

A stem map showing:
- **Green circles** — Scan A tree positions (reference)
- **Orange triangles** — Scan B tree positions (rotated to match)
- **Gray lines** — Connections between matched tree pairs
- **Rotation and translation** — Displayed in the bottom-left corner

### scan_match_report.csv

| Column | Description |
|--------|-------------|
| TreeId_A | Tree ID from scan A |
| TreeId_B | Tree ID from scan B |
| Position_Offset_m | Distance between matched positions after alignment |
| DBH_A, DBH_B | DBH from each scan |
| DBH_Diff | DBH difference (B minus A) |
| Height_A, Height_B | Height from each scan |
| x_base_A, y_base_A | Tree base position in scan A coordinates |
| x_base_B, y_base_B | Tree base position in scan B coordinates (original, not rotated) |

## Interpreting Results

- **Matches with no rotation** tells you whether the scans are already aligned. If this number is high (close to the total tree count), the scans share a coordinate system and no rotation is needed.
- **Best rotation** is the angle scan B must be rotated (counterclockwise) to match scan A. Values near 0° or 360° mean the scans are already nearly aligned. Values near 180° mean one scanner was facing the opposite direction.
- **Translation** is usually small (< 1m) if both scans are centered on the same plot. Large translation values may indicate the scans cover different areas.
- **Match distance** shows how well individual trees line up after alignment. Mean distances under 0.5m indicate good agreement.
- **DBH comparison** reveals measurement consistency between scanners or scan sessions.

## Tips

- The tool works with any two scans of the same plot — different scanners, different dates, different SLAM software. As long as both were processed through the Understory pipeline, it can align them.
- If you get few matches, try increasing `--match-radius` to 3.0 or higher. Dense plots with closely-spaced trees may need a smaller radius to avoid false matches.
- Tree IDs between scans will NOT correspond (tree 5 in scan A is not the same tree as tree 5 in scan B). The match report maps the correct correspondences.
- To avoid coordinate system issues in the first place: when aligning scans in CloudCompare, verify the transform is saved by exporting both clouds, then re-importing them into a fresh session to confirm they're still aligned.
