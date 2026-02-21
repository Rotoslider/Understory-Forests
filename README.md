# Understory Forests

### Forest Structural Complexity Tool

Understory Forests is a desktop application for extracting plot-scale forest measurements from high-resolution point clouds. It works with Terrestrial Laser Scanning (TLS), Mobile Laser Scanning (MLS), Terrestrial Photogrammetry, and UAS Photogrammetry data.

Built on top of the FSCT pipeline by Sean Krisanski, Understory provides a full GUI workflow: load a point cloud, prepare it, run semantic segmentation and measurement, then explore results in an interactive 3D viewer with branded reports and PDF export.

---

## Features

### Interactive 3D Point Cloud Viewer

- PyVista-based rendering with Level-of-Detail (automatic downsampling at 1M / 5M / 20M point thresholds for smooth interaction)
- Eye-Dome Lighting for depth perception on point clouds without normals
- Camera presets: Top, Front, Right, Isometric views (Ctrl+1 through Ctrl+4)
- Multiple color modes: RGB, Height, Classification (semantic labels), and Tree ID
- Click-to-focus point picking
- Interactive draggable plot circle in 3D for setting plot centre and radius
- Screenshot export (Ctrl+Shift+E) — save the current view as PNG, JPEG, or TIFF at 2x resolution
- Classification legend with labeled color swatches (Terrain, Vegetation, CWD, Stem)
- Cross-section slicing — horizontal or vertical slice through the point cloud for inspecting internal structure
- Measurement tools — click two points to measure 3D distance or vertical height difference
- Point cloud comparison — load a second cloud and visualize nearest-neighbor distances with a diverging colormap

### Point Cloud Preparation

- **Axis swap and rotation** — Fix orientation when Z doesn't point up (Y-Z, X-Z, X-Y swaps + 90-degree Z rotation)
- **Outlier cropping** — Remove points beyond the 99.5th percentile per axis (actually removes points from data, not just visual hiding)
- **Voxel-grid subsampling** — Reduce point density before processing (configurable spacing)
- **Plot circle visualization** — See and drag the plot boundary in 3D before running the pipeline
- **Save prepared cloud** — Export modified point cloud as .las for future use
- **Undo/Redo** — 5-level undo history (Ctrl+Z / Ctrl+Shift+Z) for axis swap, crop, and subsample operations

### Processing Pipeline

Runs all FSCT stages with real-time progress tracking and error translation:

1. **Preprocessing** — Trims point cloud to plot radius, generates box samples for inference
2. **Semantic Segmentation** — PointNet++ deep learning model classifies points into 4 classes: terrain, vegetation, coarse woody debris, and stems/branches
3. **Post-processing** — Generates DTM, separates point classes, cleans segmentation
4. **Measurement** — Cylinder fitting, tree detection, DBH/height/volume extraction, taper profiles
5. **Report Generation** — Branded HTML report with automatic PDF export

Each stage can be enabled/disabled independently. The pipeline runs in a background thread so the GUI stays responsive.

The pipeline supports **cooperative cancellation** — clicking Stop signals a safe shutdown at the next stage boundary rather than hard-killing the worker thread. A **dual progress bar** shows both overall pipeline progress and current stage progress, with sub-stage tracking during semantic segmentation.

### Tree-Aware Plot Cropping

When a plot radius is set, the pipeline uses tree-aware cropping to avoid cutting boundary trees in half. Trees whose base falls within the plot radius are fully included, even if their canopy extends beyond. A configurable buffer (default 0.5m when not specified) controls the processing boundary.

### GPU Live Monitoring

The status bar shows real-time GPU utilization and memory usage during pipeline execution, updated every 2 seconds via nvidia-smi. Falls back to PyTorch memory reporting when nvidia-smi is unavailable.

### Project Management

- **YAML-based project files** — All settings saved in a single `project.yaml` with sensible defaults
- **Timestamped run folders** — Each pipeline run creates a `runs/run_YYYY-MM-DD_HH-MM-SS/` folder preserving its output, report, and config snapshot
- **Prepared cloud tracking** — Projects remember whether a prepared (subsampled/cropped) cloud should be used for processing
- **Recent projects** — File > Recent Projects shows up to 10 recently opened files/projects
- **Drag-and-drop** — Drop .las, .laz, .pcd, or .yaml files directly onto the main window to open them
- **Field photo attachment** — Attach site photos to the project; photos appear in the HTML report

Project folder structure:
```
MyProject/
  project.yaml              # Master project configuration
  tree_registry.json         # Persistent tree IDs across runs
  runs/
    run_2026-02-15_14-30-00/
      run_config.yaml        # Settings snapshot for this run
      output/                # Pipeline .las and .csv outputs
      reports/               # HTML report, PDF, stem map, histograms
    run_2026-02-16_09-15-00/
      ...
```

### Run History

The Results tab includes a **Pipeline Run History** selector showing all previous runs for the current project. Selecting a run refreshes the output layer checkboxes, tree measurement table, and report buttons so you can browse and compare results from different pipeline runs or parameter settings.

### Persistent Tree IDs

The tree registry (`tree_registry.json`) maintains consistent tree numbering across repeated scans and pipeline runs of the same plot. After each measurement stage, trees are spatially matched against the registry using KD-tree lookups (2m match radius) with DBH similarity as a tiebreaker. New trees get sequential IDs; previously seen trees keep their existing ID. This enables meaningful comparison of tree measurements over time.

### Reports

Pipeline runs generate a branded HTML report with the Understory green theme, including:

- **Project metadata** — Filename, project name, operator, date, plot geometry
- **Summary statistics** — Tree count, stems/ha, mean DBH, mean height, canopy cover, total volume
- **Point cloud statistics** — Original, trimmed, and subsampled point counts; terrain, vegetation, CWD, and stem point breakdowns
- **Coverage and terrain** — Understory vegetation coverage, CWD coverage, average gradient
- **Tree measurements table** — Per-tree DBH, height, volume, CCI, base coordinates
- **DBH distribution** — Min, median, max with histogram
- **Stem map** — Contour map with tree positions and ID labels
- **Distribution histograms** — DBH, height, Volume 1, Volume 2
- **Processing details** — Per-stage timing in minutes and seconds
- **Notes** — User-provided project notes
- **Stand metrics** — Basal area (m²/ha), Quadratic Mean Diameter, Lorey's Mean Height, Stand Density Index
- **Taper profile charts** — Diameter vs height curves for each measured tree
- **Crown projection map** — Bird's-eye view of crown positions and sizes overlaid on DTM contours
- **Field photos** — Site photos attached to the project, displayed as a thumbnail grid

Reports are saved to the run's `reports/` folder. The output/ folder contains only pipeline data files (.las, .csv).

### PDF Export

One-click PDF export from the HTML report using Qt's WebEngine renderer. Generates A4 portrait PDFs with 10mm margins, saved to the run's reports folder. No additional dependencies required.

### Output Layer Viewer

The Results tab provides checkboxes for 19 output point cloud layers. Select any combination and load them into the 3D viewer with semantic classification coloring:

| Layer | File | Description |
|-------|------|-------------|
| DTM | DTM.las | Digital Terrain Model |
| Cropped DTM | cropped_DTM.las | DTM cropped to plot radius |
| Terrain Points | terrain_points.las | Ground/terrain classified points |
| Vegetation Points | vegetation_points.las | Vegetation classified points |
| CWD Points | cwd_points.las | Coarse Woody Debris points |
| Stem Points | stem_points.las | Tree stem classified points |
| Ground Vegetation | ground_veg.las | Vegetation below canopy cutoff |
| Segmented | segmented.las | Full cloud with semantic labels + confidence scores |
| Segmented Cleaned | segmented_cleaned.las | Cleaned segmentation output |
| Stem Points Sorted | stem_points_sorted.las | Stem points assigned to trees |
| Veg Points Sorted | veg_points_sorted.las | Vegetation assigned to trees |
| Skeleton Clusters | skeleton_cluster_visualisation.las | Tree skeleton visualization |
| Cylinder Model | full_cyl_array.las | Fitted cylinder array |
| Sorted Cylinders | sorted_full_cyl_array.las | Cylinders sorted by tree ID |
| Cleaned Cylinders | cleaned_cyls.las | Filtered cylinder measurements |
| Cleaned Cyl Vis | cleaned_cyl_vis.las | Cleaned cylinder visualization |
| Interpolated Cylinders | interpolated_full_cyl_array.las | Gap-filled cylinder array |
| Text Labels | text_point_cloud.las | 3D tree ID labels |
| Tree-Aware Crop | tree_aware_cropped_point_cloud.las | Plot-cropped output cloud |

Clicking a tree row in the measurements table highlights that tree in the viewer and focuses the camera on it.

### Tree Data Export

Tree measurement data can be exported as CSV from the Results tab. The save dialog defaults to the run's reports folder for organized output.

### Batch Processing

Process multiple point clouds in one go. Open **Tools > Batch Processing**, add files, and run them all using the current pipeline settings. Each file gets its own run folder. GPU memory is automatically cleaned between runs.

### Run Comparison Report

Compare results from two pipeline runs with **Tools > Compare Runs**. The comparison report shows per-tree deltas for DBH, height, and volume, identifies newly detected or missing trees, and generates a standalone HTML report with change tables and delta histograms.

### Growth Dashboard

Track tree growth over time with **Tools > Growth Dashboard**. Select trees from the registry to see DBH and height plotted across multiple scans. Export growth data as CSV for external analysis.

### Allometric Equations

Apply biomass and carbon equations to your tree data with **Tools > Allometric Equations**. Understory includes default generic AGB and carbon equations and lets you define custom formulas using any column from tree_data.csv. Equations are evaluated against all trees with results displayed in a table and exportable as CSV.

### GIS Export

Export tree locations and attributes for use in GIS software. The **Export to GIS** button in the Results tab writes GeoJSON (no extra dependencies) or Shapefile (requires geopandas). Each tree is a Point feature with all measurements as attributes. Specify a CRS string for georeferenced output.

### Flythrough Animation

Create camera flythrough animations with **View > Flythrough Editor**. Capture keyframe positions from the current 3D view, then render a smooth camera path using cubic spline interpolation. Export as an image sequence, GIF, or MP4 (MP4 requires imageio-ffmpeg).

### Training Workflow

A dedicated Training panel provides a guided 5-step workflow for retraining the PointNet++ model on your own data:

1. **Import Training Data** — Import labeled .las files into the training directory
2. **Bootstrap Labels** — Run the pipeline on unlabeled data to generate initial labels automatically
3. **Review & Correct Labels** — Open the Label Editor to fix mistakes, using confidence highlighting to find uncertain areas
4. **Configure Training** — Set epochs, learning rate, batch sizes, device, and class weights with tooltips explaining each parameter
5. **Train Model** — Run training with progress tracking (epoch, loss, accuracy)

Key training features:
- **Class weights** — Auto-compute inverse-frequency weights so underrepresented classes (like CWD) get higher training weight, significantly improving their accuracy
- **Confidence-guided correction** — The Label Editor shows model confidence per point, letting you focus editing time on the points the model is least sure about
- **Fine-tuning support** — Load existing model weights and fine-tune on new site-specific data instead of training from scratch

### Label Editor

A full-featured point cloud label editor for correcting semantic segmentation labels and preparing training data:

- **Box selection** — Drag a rectangle to select points in the 3D view
- **Class painting** — Assign selected points to any of the 4 semantic classes (keyboard shortcuts 1-4)
- **Undo/Redo** — Full undo/redo history (Ctrl+Z / Ctrl+Shift+Z)
- **Layer visibility** — Toggle individual classes on/off with checkboxes to isolate the layer you're editing
- **Set Focus** — Right-click to set the camera orbit point (F key toggle), making it easy to navigate around specific areas
- **Eye-Dome Lighting** — Toggle EDL for better depth perception on dense clouds
- **Camera presets** — Top, Front, Right, Isometric views plus Reset View (Home key)
- **Confidence visualization** — Color points by model confidence (red = low, green = high) to quickly find uncertain areas that need correction (C key toggle)
- **Select Low Confidence** — One-click selection of all points below a confidence threshold for batch correction
- **Save Labels** — Export corrected labels as .las files for training
- **Auto label detection** — Automatically detects 0-indexed vs 1-indexed labels and adjusts accordingly

### Per-Run Settings Persistence

Each pipeline run saves a snapshot of all settings used (`run_config.yaml`). When you select a previous run in the Results tab, all parameter fields are automatically populated with the settings from that run. This makes it easy to compare what changed between runs and to re-run with the same or slightly modified settings.

### Right-Click Reset to Default

Every parameter field in the Process and Advanced tabs supports right-click to reset to its default value. The context menu shows the default so you know what you're resetting to.

### GPU Memory Management

GPU memory is automatically released after the semantic segmentation stage completes — not just when the application closes. This means you can run the pipeline, review results, and continue using the GPU for other tasks without restarting the app. A safety cleanup also runs if the pipeline encounters an error.

### Configuration

All pipeline parameters are exposed through the GUI with tooltips explaining each setting:

**Process Tab:**
- Plot radius and buffer
- Plot centre (auto or manual coordinates)
- Model selection and import
- Batch size, CPU cores, CPU-only mode
- Pipeline stage toggles

**Advanced Tab:**
- DTM grid resolution
- Cylinder fitting: minimum CCI, min tree cylinders, min cluster size
- Height percentile, tree base cutoff, ground vegetation cutoff
- Vegetation and stem sorting ranges
- Slice thickness and increment
- Taper measurement heights and increments
- Segmentation box dimensions and overlap
- Working directory cleanup, minimise output mode

---

## Installation

### System Requirements

- **OS:** Ubuntu 24.04 (primary platform)
- **Python:** 3.10 – 3.12 (developed on 3.12.3)
- **GPU:** NVIDIA GPU with CUDA support (strongly recommended)
- **NVIDIA Driver:** 570+ with CUDA 12.8
- **RAM:** 16+ GB (128 GB recommended for large point clouds)

### Quick Install (Recommended)

The easiest way to set up Understory on a fresh Ubuntu system:

```bash
git clone <repo-url> Understory_Forests
cd Understory_Forests
chmod +x install.sh
./install.sh
```

The script checks prerequisites, creates a virtual environment, installs PyTorch with CUDA 12.8, installs Understory in editable mode, and builds the PyG extensions.

### Manual Setup

```bash
cd Understory_Forests

# Create and activate virtual environment
sudo apt install python3.12-venv
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
sudo apt-get install -y build-essential python3-dev

# Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Understory in editable mode
pip install -e .

# Install PyTorch Geometric extensions (requires torch at build time)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.10.0+cu128.html --no-build-isolation
```

If you do not have an NVIDIA GPU, check the "CPU Only" option in the GUI Process tab.

---

## How to Use

### Launch the GUI

```bash
source venv/bin/activate
python -m understory
```

### Workflow

1. **Project tab** — Browse to select your .las, .laz, or .pcd point cloud file. Enter a project name, operator, and notes.
2. **Prepare tab** — Fix orientation if needed (axis swap), crop outliers, preview subsampling. Save the prepared cloud for reproducibility.
3. **Process tab** — Set plot radius, choose which pipeline stages to run, select the model, adjust batch size. Click **Run Pipeline**.
4. **Advanced tab** — Fine-tune DTM resolution, cylinder fitting thresholds, taper measurement parameters, and segmentation box settings.
5. **Results tab** — Browse previous runs, load output layers into the viewer, view tree measurements, open the HTML report, or export a PDF.

### Keyboard Shortcuts

**Main Window:**

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New project |
| Ctrl+O | Open point cloud |
| Ctrl+Shift+O | Open project |
| Ctrl+S | Save project |
| Ctrl+Shift+S | Save project as |
| Ctrl+W | Close point cloud |
| F5 | Run pipeline |
| Shift+F5 | Stop pipeline |
| Home | Reset camera |
| Ctrl+1/2/3/4 | Top/Front/Right/Isometric view |
| Ctrl+Shift+E | Export screenshot |
| Ctrl+Z | Undo (Prepare tab) |
| Ctrl+Shift+Z | Redo (Prepare tab) |
| Escape | Cancel measurement |
| Ctrl+Q | Exit |

**Label Editor:**

| Shortcut | Action |
|----------|--------|
| 1/2/3/4 | Quick-assign selected points to Terrain/Vegetation/CWD/Stem |
| F | Toggle Set Focus mode (right-click to pick focus point) |
| C | Toggle confidence coloring |
| Home | Reset camera view |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z | Redo |

### User Guide

See the **[User Guide](USER_GUIDE.md)** ([PDF](USER_GUIDE.pdf)) for a complete walkthrough of every tab and setting in Understory — from creating projects and preparing point clouds, to understanding advanced parameters and exporting results. Written in clear, simple language.

### Training Your Own Model

See the **[Training Guide](TRAINING_GUIDE.md)** ([PDF](TRAINING_GUIDE.pdf)) for a step-by-step tutorial on preparing training data and retraining the model for your specific forest type. Covers the full workflow from raw scan to improved model.

---

## Data Outputs

### CSV Files

- **tree_data.csv** — Per-tree measurements: DBH, height, volume (two methods), CCI, crown position, base coordinates
- **taper_data.csv** — Diameter at height measurements for each stem
- **plot_summary.csv** — Plot-level statistics (point counts, coverage fractions, timing, tree summaries)

### tree_data.csv Columns

| Column | Description |
|--------|-------------|
| TreeId | Persistent tree identifier (consistent across runs via tree registry) |
| x/y/z_tree_base | Tree base coordinates |
| DBH | Diameter at breast height (1.3m above ground) |
| CCI_at_BH | Circumferential Completeness Index — fraction of circle with point coverage |
| Height | Tree height |
| Volume_1 | Sum of fitted cylinder volumes |
| Volume_2 | Cone + cylinder approximation |
| Crown_mean_x/y | Crown centroid |
| Crown_top_x/y/z | Highest crown point |
| Crown_area | Crown projection area (m²) |
| mean_understory_height | Average understory height within 5m radius |

**CCI** indicates scan completeness: single-scan TLS typically maxes at ~0.5 (one side visible), complete multi-scan coverage approaches 1.0.

---

## Semantic Classes

The PointNet++ model classifies points into 4 classes:

| Class | ID | Description |
|-------|----|-------------|
| Terrain | 1 | Ground surface |
| Vegetation | 2 | Leaves, branches, understory |
| CWD | 3 | Coarse Woody Debris |
| Stems | 4 | Tree trunks and major branches |

---

## Project Structure

```
Understory_Forests/
├── understory/              # Understory GUI package
│   ├── __main__.py          # Entry point (python -m understory)
│   ├── core/                # Pipeline, paths, reports, tree registry
│   │   ├── pipeline.py      # Pipeline runner with cooperative cancellation
│   │   ├── paths.py         # Project and output path management
│   │   ├── report.py        # Jinja2 HTML report + PDF export
│   │   ├── tree_registry.py # Persistent tree ID matching
│   │   ├── comparison.py    # Multi-run comparison reports
│   │   ├── allometry.py     # Allometric equation evaluation
│   │   └── gis_export.py    # GeoJSON and Shapefile export
│   ├── config/              # Dataclass-based configuration
│   │   └── settings.py      # ProjectConfig with YAML serialization
│   ├── gui/                 # PySide6 interface
│   │   ├── main_window.py   # Main window, menus, GPU monitor
│   │   ├── panels/          # Sidebar panels (processing, training, batch, growth, allometry)
│   │   └── viewer/          # 3D point cloud viewer, label editor
│   └── resources/           # Icons, QSS stylesheets, report template
├── scripts/                 # Original FSCT pipeline
│   ├── preprocessing.py     # Point cloud → box samples
│   ├── inference.py         # PointNet++ semantic segmentation
│   ├── post_segmentation_script.py  # DTM + point cleaning
│   ├── measure.py           # Tree measurements + cylinder fitting
│   ├── model.py             # PointNet++ architecture (4 classes)
│   └── tools.py             # File I/O, clustering, utilities
├── model/                   # Trained model weights (model.pth)
├── tests/                   # Test suite (76 tests)
├── data/                    # Training/test datasets
├── install.sh               # Automated setup script
└── pyproject.toml           # Package configuration
```

---

## Testing

```bash
# Fast tests (~5 seconds, 73 tests)
./venv/bin/python -m pytest tests/ -m "not slow" -v

# All tests including end-to-end pipeline (~40 minutes, 76 tests)
./venv/bin/python -m pytest tests/ -v
```

Test coverage spans: configuration roundtrips, path construction, file I/O (LAS/PCD), GPU detection, PyTorch Geometric operations, model architecture, report template rendering, tree registry matching, and pipeline integration.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| GUI Framework | PySide6 (Qt 6.10) |
| 3D Rendering | PyVista + pyvistaqt |
| PDF Export | QtWebEngineWidgets |
| Deep Learning | PyTorch 2.10 + PyTorch Geometric 2.7 |
| Point Cloud I/O | laspy (LAS/LAZ), open3d (PCD) |
| Configuration | Python dataclasses + PyYAML |
| Reports | Jinja2 HTML templates |
| Testing | pytest |

Developed and tested on:
- Ubuntu Linux (kernel 6.17)
- NVIDIA RTX PRO 6000 (95 GB VRAM), CUDA 12.8
- Python 3.12.3

---

## Performance

The measurement stage (`measure.py`) has been optimized to use list-append + single `np.vstack()` instead of repeated `np.vstack()` inside loops. This changes ~45 instances from O(n^2) to O(n) memory allocation, significantly improving performance on large point clouds.

GPU memory is released immediately after semantic segmentation completes, rather than being held until the application closes.

---

## Known Limitations

- Young trees with heavy branching may not segment correctly
- Extremely large trees may not measure properly
- Low-resolution point clouds (below high-res ALS quality) produce poor results
- Small branches are often missed
- Completely horizontal branches may not measure correctly
- The `low_resolution_point_cloud_hack_mode` setting can help borderline datasets
- CPU-only mode is significantly slower and may produce slightly different segmentation results

---

## Citation

If citing in a scientific journal:

> Krisanski, S.; Taskhiri, M.S.; Gonzalez Aracil, S.; Herries, D.; Muneri, A.; Gurung, M.B.; Montgomery, J.; Turner, P. Forest Structural Complexity Tool — An Open Source, Fully-Automated Tool for Measuring Forest Point Clouds. Remote Sens. 2021, 13, 4677. https://doi.org/10.3390/rs13224677

Otherwise, link to the GitHub repository.

## License & Use

This code is free to use, modify, and share, including for commercial purposes. Contributions and improvements are welcome.

## Acknowledgements

Original FSCT research was funded by the Australian Research Council — Training Centre for Forest Value (IC150100004), University of Tasmania, Australia.

Thanks to the supervisory team: Assoc. Prof Paul Turner, Dr. Mohammad Sadegh Taskhiri, and Dr. James Montgomery. Thanks to Susana Gonzalez Aracil and David Herries (Interpine Group Ltd, NZ), Allie Muneri and Mohan Gurung (PF Olsen Australia Ltd) for raw point clouds and plot measurements.

## References

- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [PointNet++](https://github.com/charlesq34/pointnet2) (original), [PyG implementation](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py)
- [PyVista](https://docs.pyvista.org/)
- [PySide6](https://doc.qt.io/qtforpython-6/)
