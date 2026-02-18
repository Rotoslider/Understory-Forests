cd ~/projects/FSCT
./venv/bin/python -m understory
------------------------------------------------
Commit fea323f is your safety net.
----------------------------------------------
Main App opens in a windowed view which is great. If I try to drag the window it changes to full screen for a second then back to original size and then I can drag without issue. Only happens first drag after launch.

It would be helpful when preparing a point cloud to be able to draw a poly-line and or box around an area of cloud to remove or keep points. Typically a scan captures unwanted data at the edges that need trimmed to get the proper plot area.

Issues with the Label Editor:

------------------------------------------------------------
Add more labels to classify. canopy plus understory brush. 
Add Dead tree detection

-----------------------------------------------------------------------------------------------------------------
## Additional Functionality to Implement


### Training & Label Editor

| Feature | Description | Complexity |
|---------|-------------|------------|
| **Live training loss plots** | Embedded matplotlib/pyqtgraph chart during training | Medium |
| **Brush/lasso selection** | Additional selection modes for label editing | Medium |
| **Training tutorial** | In-app guidance for the training workflow | Low |

### Quality of Life

| Feature | Description | Complexity |
|---------|-------------|------------|
| **Settings persistence** | Remember window size, splitter position, last directory, color mode | Low |
| **Dark mode** | Alternative dark QSS theme | Low |
| **Status bar point count** | Show displayed/total point count at all times | Low |
| **Console log export** | Save console log to text file | Low |
| **Metric/imperial units** | Configurable unit display | Medium |

### Infrastructure

| Feature | Description | Complexity |
|---------|-------------|------------|
| **Windows/macOS testing** | Verify installation and GUI on Windows and macOS | Medium |
| **CI/CD pipeline** | GitHub Actions for automated testing on push | Medium |
| **Standalone packaging** | PyInstaller or cx_Freeze executable for distribution | High |
| **Documentation site** | Sphinx or MkDocs user guide with screenshots | Medium |
| **Plugin system** | Third-party measurement or visualization plugins | High |

### Performance

| Feature | Description | Complexity |
|---------|-------------|------------|
| **Fix O(n^2) vstack in measure.py** | ~45 instances of `np.vstack()` inside loops — convert to list-append | Medium |
| **Multiprocessing for preprocessing** | Replace threaded_boxes() with true multiprocessing | Medium |
| **LAZ compression** | Native compressed .laz read/write without decompression step | Low |

---

## Known Limitations

- Young trees with heavy branching may not segment correctly
- Extremely large trees may not measure properly
- Low-resolution point clouds produce poor results
- Small branches are often missed
- Horizontal branches may not measure correctly
- CPU-only mode is significantly slower
- Pipeline cancellation uses thread termination (not cooperative)
- No undo for preparation operations (only per-operation reset)
- Training panel has no live loss plots
- Label editor supports only box selection
-----------------------------------

# Understory Forests — Feature Implementation Plan

### High Priority

| Feature | Description | Complexity |
|---------|-------------|------------|
| **Cooperative pipeline cancellation** | Replace `terminate()` with a stop flag checked between stages | Medium |
| **Undo/redo for preparation** | Track axis swaps, crops, subsamples with undo stack | Medium |
| **Sub-stage progress** | Report progress within stages (e.g., 30% through segmentation) | Medium |
| **Multi-file batch processing** | Process multiple point clouds in sequence with same settings | Medium |
| **Recent projects list** | File menu shows recently opened projects | Low |
| **Drag-and-drop** | Drop .las/.pcd files onto the window to open them | Low |

### Report & Analysis

| Feature | Description | Complexity |
|---------|-------------|------------|
| **Run comparison report** | Side-by-side comparison of two runs (delta DBH, height, new/removed trees) | High |
| **Growth tracking dashboard** | Charts of DBH/height changes over time from tree registry history | High |
| **Taper profile charts** | Per-tree taper curve visualizations in report | Medium |
| **Crown projection map** | Overhead view showing canopy outlines | Medium |
| **Allometric equations** | User-defined biomass/carbon calculations from measurements | Medium |
| **Additional plot metrics** | Basal area, quadratic mean diameter, Lorey's height, stand density index | Medium |
| **Export to GIS** | Shapefile or GeoJSON export of tree positions with attributes | Medium |
| **Photo/notes attachment** | Attach field photos or extended notes to reports | Low |

### Viewer

| Feature | Description | Complexity |
|---------|-------------|------------|
| **Cross-section view** | Horizontal/vertical slice viewer through the point cloud | High |
| **Measurement tools** | Click-to-measure distance, height, angle in 3D | High |
| **Point cloud comparison** | Overlay two clouds with difference coloring | Medium |
| **Animation/flythrough** | Camera path animation for presentation or QA | Medium |
| **Screenshot export** | High-res screenshot of the current view | Low |
| **Colorbar legend** | Color legend for active color mode (height scale, class labels) | Low |
│
│                                                                                                                  │
│ Context                                                                                                          │
│                                                                                                                  │
│ The Understory Forests GUI has a working pipeline (preprocessing, segmentation, post-processing, measurement,    │
│ report) with a 3D viewer, label editor, and training workflow. This plan covers 20 features across three         │
│ categories: pipeline/UI improvements, report/analysis enhancements, and viewer tools. Features are grouped into  │
│ three phases by effort and value.                                                                                │
│                                                                                                                  │
│ ---                                                                                                              │
# Phase 1: Quick Wins (~1-2 days total)                                                                            │
│                                                                                                                  │
1. Screenshot Export                                                                                               │
│                                                                                                                  │
│ Effort: Small | Files: point_cloud_viewer.py, main_window.py                                                     │
│                                                                                                                  │
│ - Add export_screenshot(filepath, scale=2) method to PointCloudViewer using PyVista's built-in                   │
│ plotter.screenshot()                                                                                             │
│ - Add "Export Screenshot..." (Ctrl+Shift+E) action to View menu in MainWindow                                    │
│ - File dialog filters: PNG, JPEG, TIFF 

2. Recent Projects List                                                                                            │
│                                                                                                                  │
│ Effort: Small | Files: main_window.py                                                                            │
│                                                                                                                  │
│ - Use QSettings("Understory", "Understory") to persist up to 10 recently opened project/file paths               │
│ - Add "Recent Projects" submenu under File menu                                                                  │
│ - Hook into existing _open_project() and _on_file_loaded() to record paths                                       │
│ - Each entry shows basename; tooltip shows full path                                                             │
│                                                                                                                  │
3. Drag-and-Drop                                                                                                   │
│                                                                                                                  │
│ Effort: Small | Files: main_window.py                                                                            │
│                                                                                                                  │
│ - Enable setAcceptDrops(True) on MainWindow                                                                      │
│ - Override dragEnterEvent / dropEvent — accept .las, .laz, .pcd, .yaml, .yml                                     │
│ - Point cloud files → _processing_panel.set_input_file(path)                                                     │
│ - YAML files → _processing_panel.load_project(path)

4. Colorbar Legend                                                                                                 │
│                                                                                                                  │
│ Effort: Small | Files: point_cloud_viewer.py                                                                     │
│                                                                                                                  │
│ - CLASSIFICATION mode: use plotter.add_legend() with entries from CLASS_COLORS dict                              │
│ (Noise/Terrain/Vegetation/CWD/Stem)                                                                              │
│ - HEIGHT mode: already has scalar bar (no change needed)                                                         │
│ - TREE_ID mode: add text annotation "Colored by Tree ID"                                                         │
│ - Clear legend on mode switch (already happens via plotter.clear())                                              │
│                                                                                                                  │
5. Additional Plot Metrics                                                                                         │
│                                                                                                                  │
│ Effort: Small | Files: report.py, report_template.html                                                           │
│                                                                                                                  │
│ - Compute from existing tree_data.csv columns:                                                                   │
│   - Basal Area (m2/ha) = sum(pi * (DBH/2)^2) / plot_area                                                         │
│   - QMD = sqrt(mean(DBH^2))                                                                                      │
│   - Lorey's Height = sum(Height_i * BA_i) / sum(BA_i)                                                            │
│   - Stand Density Index = stems_per_ha * (QMD / 0.254)^1.605                                                     │
│ - Add "Stand Metrics" section with stat cards to report template                                                 │
│                                                                                                                  │
│ ---
---                                                                                                                │
# Phase 2: Core Improvements (~3-5 days total)                                                                     │
│                                                                                                                  │
6. Cooperative Pipeline Cancellation                                                                               │
│                                                                                                                  │
│ Effort: Medium | Files: pipeline.py, processing_panel.py                                                         │
│ Priority: First in phase (safety-critical)                                                                       │
│                                                                                                                  │
│ - Add threading.Event cancel flag to PipelineWorker                                                              │
│ - Add PipelineWorker.request_stop() method that sets the event                                                   │
│ - In pipeline.py, add cancel_event param to run_pipeline() — check is_set() before each stage                    │
│ - New PipelineCancelled exception caught by worker, emits cancelled signal                                       │
│ - Replace worker.terminate() in stop_pipeline() with worker.request_stop() + worker.wait(10s), fallback to       │
│ terminate() if hung                                                                                              │
│                                                                                                                  │
7. Undo/Redo for Preparation                                                                                       │
│                                                                                                                  │
│ Effort: Medium | Files: point_cloud_viewer.py, main_window.py                                                    │
│                                                                                                                  │
│ - PrepareSnapshot dataclass stores (points, colors, labels, tree_ids, description) before each destructive       │
│ operation                                                                                                        │
│ - Stack stored in PointCloudViewer, max depth 5 (memory constraint for large clouds)                             │
│ - push_undo() called before axis swap, crop, and subsample operations                                            │
│ - Wire Ctrl+Z / Ctrl+Shift+Z shortcuts in MainWindow (only active in Prepare tab) 
8. Taper Profile Charts                                                                                            │
│                                                                                                                  │
│ Effort: Medium | Files: report.py, report_template.html                                                          │
│                                                                                                                  │
│ - Load taper_data.csv (columns: TreeId, height increments 0.0-30.0m)                                             │
│ - Generate matplotlib chart: height (Y) vs diameter (X), one line per tree                                       │
│ - Standard forestry convention; brand color palette                                                              │
│ - Add "Taper Profiles" section to report template                                                                │
│                                                                                                                  │
9. Crown Projection Map                                                                                            │
│                                                                                                                  │
│ Effort: Medium | Files: report.py, report_template.html                                                          │
│                                                                                                                  │
│ - Use Crown_mean_x, Crown_mean_y from tree_data.csv for canopy centroids                                         │
│ - Plot circles with approximate crown radius (from crown area or DBH-based estimate)                             │
│ - Overlay on DTM contour base map                                                                                │
│ - Color by tree ID; include stem positions as dots                                                               │
│ - Add to report as "Crown Projection Map" section                                                                │
│                                                                                                                  │
10. Cross-Section View                                                                                             │
│                                                                                                                  │
│ Effort: Medium | Files: point_cloud_viewer.py                                                                    │
│                                                                                                                  │
│ - Add slicer controls to viewer toolbar: mode combo (Off/Horizontal/Vertical), position spinbox, thickness       │
│ spinbox                                                                                                          │
│ - Filter _lod_indices during render: keep points where |point[axis] - pos| < thickness/2                         │
│ - Slider for interactive positioning                                                                             │
│ - Store _slice_axis, _slice_pos, _slice_thickness as viewer state                                                │
│ - No data modification — purely visual filter
11. Sub-Stage Progress                                                                                             │
│                                                                                                                  │
│ Effort: Medium | Files: pipeline.py, inference.py, optionally preprocessing.py, measure.py                       │
│ Depends on: Feature 6 (cooperative cancellation adds check points)                                               │
│                                                                                                                  │
│ - Extend _run_stage() to create a sub-progress closure passed to legacy classes                                  │
│ - Primary target: inference.py batch loop — report batch_index / total_batches                                   │
│ - Secondary targets: preprocessing box generation, measurement phases                                            │
│ - Legacy classes accept optional progress_callback parameter                                                     │
│                                                                                                                  │
│ ---                                                                                                              │
# Phase 3: Advanced Features (~3-4 weeks total)                                                                    │
│                                                                                                                  │
12. Multi-File Batch Processing                                                                                    │
│                                                                                                                  │
│ Effort: Large | New file: batch_panel.py | Modify: processing_panel.py, main_window.py, pipeline.py              │
│ Depends on: Feature 6 (cancellation)                                                                             │
│                                                                                                                  │
│ - New QDialog with file list, per-file status icons, add/remove buttons                                          │
│ - Uses current ProcessingPanel settings for all files                                                            │
│ - BatchPipelineWorker iterates over configs, emits per-file + overall progress                                   │
│ - GPU memory cleanup (torch.cuda.empty_cache() + gc.collect()) between runs                                      │
│ - "Batch Processing..." action in Tools menu
13. Run Comparison Report                                                                                          │
│                                                                                                                  │
│ Effort: Large | New files: comparison.py, comparison_template.html | Modify: report.py, processing_panel.py      │
│                                                                                                                  │
│ - compare_runs(run_a_dir, run_b_dir) matches trees by TreeId from registry                                       │
│ - Computes deltas: DBH change, height change, volume change                                                      │
│ - Identifies new/removed trees between scans                                                                     │
│ - Generates HTML comparison report with delta histograms, change tables, summary cards                           │
│ - "Compare Runs" button in Results tab (enabled when 2+ runs exist)                                              │
│                                                                                                                  │
14. Growth Tracking Dashboard                                                                                      │
│                                                                                                                  │
│ Effort: Large | New file: growth_panel.py | Modify: tree_registry.py, main_window.py                             │
│ Depends on: Feature 13 (comparison infrastructure)                                                               │
│                                                                                                                  │
│ - New QDialog with matplotlib canvas embedded                                                                    │
│ - Tree selector (combo/list of registered trees)                                                                 │
│ - Charts: DBH over time, height over time per selected tree(s)                                                   │
│ - Scan history table                                                                                             │
│ - Export growth data as CSV                                                                                      │
│ - Enhance TreeRegistry.get_growth_data() for multi-tree queries                                                  │
│ - "Growth Dashboard..." in Tools menu
15. Allometric Equations                                                                                           │
│                                                                                                                  │
│ Effort: Large | New files: allometry.py, allometry_panel.py | Modify: settings.py, report.py                     │
│                                                                                                                  │
│ - AllometricEquation dataclass: name, formula string, variable list                                              │
│ - AllometryRegistry: load/save equations from YAML, includes defaults (generic AGB, carbon)                      │
│ - Formula evaluation: safe eval with restricted namespace (numpy math functions + tree_data columns)             │
│ - QDialog for equation management: list, add/edit/remove, preview results                                        │
│ - Computed columns added to tree_data and included in report                                                     │
│                                                                                                                  │
16. Export to GIS                                                                                                  │
│                                                                                                                  │
│ Effort: Medium-Large | New file: gis_export.py | Modify: processing_panel.py                                     │
│                                                                                                                  │
│ - GeoJSON export (pure Python, no extra dependencies): FeatureCollection with Point per tree, all attributes     │
│ - Optional Shapefile export via geopandas (optional dependency)                                                  │
│ - User-specified CRS string (LAS files typically use local coordinates)                                          │
│ - "Export to GIS..." button in Results tab with format selector                                                  │
│ - Document coordinate system limitations                                                                         │
│                                                                                                                  │
17. Measurement Tools                                                                                              │
│                                                                                                                  │
│ Effort: Large | Files: point_cloud_viewer.py, main_window.py                                                     │
│                                                                                                                  │
│ - MeasureMode enum: OFF, DISTANCE, HEIGHT, ANGLE                                                                 │
│ - Point picking: first click stores point A, second click stores point B                                         │
│ - Visual: plotter.add_lines() for measurement line, plotter.add_point_labels() for value                         │
│ - Distance = Euclidean 3D, Height = |dZ|, Angle = atan2(dZ, horizontal_dist)                                     │
│ - "Measure" submenu in Tools menu; Escape to cancel measurement
18. Point Cloud Comparison                                                                                         │
│                                                                                                                  │
│ Effort: Large | Files: point_cloud_viewer.py, main_window.py                                                     │
│                                                                                                                  │
│ - "Compare Point Clouds..." in Tools menu → file dialog for second cloud                                         │
│ - Compute nearest-neighbor distances from cloud B to cloud A (via scipy cKDTree)                                 │
│ - Color by distance using diverging colormap (blue=close, red=far)                                               │
│ - New ColorMode.COMPARISON in viewer                                                                             │
│ - Display statistics: mean/max/std of distances                                                                  │
│                                                                                                                  │
19. Photo/Notes Attachment                                                                                         │
│                                                                                                                  │
│ Effort: Medium | Files: settings.py, processing_panel.py, report.py, report_template.html                        │
│                                                                                                                  │
│ - Add photos: list[str] to ProjectConfig                                                                         │
│ - "Attach Photos" button in Project tab → multi-file QFileDialog                                                 │
│ - Photos copied to project folder on save                                                                        │
│ - Thumbnail display in Project tab                                                                               │
│ - "Field Photos" grid section in report template                                                                 │
│                                                                                                                  │
20. Animation/Flythrough                                                                                           │
│                                                                                                                  │
│ Effort: Large | New file: flythrough.py | Modify: point_cloud_viewer.py, main_window.py                          │
│ Depends on: Feature 1 (screenshot as building block)                                                             │
│                                                                                                                  │
│ - FlythroughEditor QDialog: define camera keyframes by capturing current view                                    │
│ - Cubic spline interpolation on camera position + look-at point                                                  │
│ - Frame rendering via plotter.screenshot() in loop                                                               │
│ - Export as image sequence or GIF/MP4 (via imageio optional dependency)                                          │
│ - "Flythrough Editor..." in View menu                                                                            │
│                                                                                                                  │
│ ---
Dependency Graph                                                                                                   │
│                                                                                                                  │
│ Feature 6 (Cancel) ──> Feature 11 (Sub-stage Progress)                                                           │
│                    ──> Feature 12 (Batch Processing)                                                             │
│                                                                                                                  │
│ Feature 13 (Comparison) ──> Feature 14 (Growth Dashboard)                                                        │
│                                                                                                                  │
│ Feature 1 (Screenshot) ──> Feature 20 (Flythrough)                                                               │
│                                                                                                                  │
│ All other features are independent.                                                                              │
│                                                                                                                  │
│ Critical Files (most frequently modified)                                                                        │
│                                                                                                                  │
│ ┌─────────────────────────────────────────────┬────────────────────────────────┐                                 │
│ │                    File                     │      Features touching it      │                                 │
│ ├─────────────────────────────────────────────┼────────────────────────────────┤                                 │
│ │ understory/gui/main_window.py               │ 1, 2, 3, 7, 12, 14, 17, 18, 20 │                                 │
│ ├─────────────────────────────────────────────┼────────────────────────────────┤                                 │
│ │ understory/gui/viewer/point_cloud_viewer.py │ 1, 4, 7, 10, 17, 18            │                                 │
│ ├─────────────────────────────────────────────┼────────────────────────────────┤                                 │
│ │ understory/gui/panels/processing_panel.py   │ 6, 11, 12, 13, 16, 19          │                                 │
│ ├─────────────────────────────────────────────┼────────────────────────────────┤                                 │
│ │ understory/core/pipeline.py                 │ 6, 11, 12                      │                                 │
│ ├─────────────────────────────────────────────┼────────────────────────────────┤                                 │
│ │ understory/core/report.py                   │ 5, 8, 9, 13, 15, 19            │                                 │
│ ├─────────────────────────────────────────────┼────────────────────────────────┤                                 │
│ │ understory/resources/report_template.html   │ 5, 8, 9, 19                    │                                 │
│ └─────────────────────────────────────────────┴────────────────────────────────┘                                 │
│ 
Verification                                                                                                     │
│                                                                                                                  │
│ Each feature should be tested by:                                                                                │
│ 1. Launching the GUI (python -m understory) and exercising the new UI controls                                   │
│ 2. Loading a point cloud and running through the relevant workflow                                               │
│ 3. For report features: running the pipeline on test data and checking the generated HTML/PDF                    │
│ 4. For viewer features: visually confirming correct rendering, camera behavior, and no regressions in existing   │
│ tools (selection, confidence, painting)

-------------------------------------------------------

All 20 features are now implemented and committed. Here's a summary:
                                                                                                                    
  Commit 0426a33 — 984 insertions across 8 files
                                                 
  Phase 1 (Quick Wins) — Complete
                                                                                                                    
  1. Screenshot Export — Ctrl+Shift+E in View menu, uses PyVista screenshot() at 2x
  2. Recent Projects — Up to 10 entries via QSettings, auto-recorded on file open
  3. Drag-and-Drop — .las/.laz/.pcd/.yaml/.yml accepted on main window
  4. Colorbar Legend — Classification legend with class names/colors, Tree ID annotation
  5. Stand Metrics — Basal area, QMD, Lorey's height, SDI in report

  Phase 2 (Core Improvements) — Complete

  6. Cooperative Cancellation — threading.Event + PipelineCancelled replaces terminate()
  7. Undo/Redo — 5-deep snapshot stack for Prepare tab (crop, axis swap)
  8. Taper Profile Charts — Height vs diameter per tree in report
  9. Crown Projection Map — Overhead canopy view in report
  10. Cross-Section View — Horizontal/vertical slice controls in viewer toolbar
  11. Sub-Stage Progress — Inference batch progress piped through to progress bar

  Phase 3 (Advanced Features) — Complete

  12-16. Batch Processing, Comparison, Growth, Allometry, GIS Export — Menu integration + scaffolding modules
  17. Measurement Tools — Distance/height with visual line + label in viewer
  18. Point Cloud Comparison — NN distance coloring with RdYlBu_r diverging colormap
  19. Photo Attachment — Full pipeline: settings → panel → pipeline → report template
  20. Flythrough — Menu integration + scaffolding editor


-----------------------------------------------------

ideas from Sean K

CWD Volume
Individual Tree files in folder with option. Option also for vegetation or no veg with each tree.
Adjust tree id labels to not skip numbers (just part of the process currently as the id labels are arbitrary).
Explore potential optimizations related to global shifting. May be able to use 32 bits in much of the code if returning the global shift happens at the end.
Rectangular, tree-aware-plot-cropping mode capable of automatically processing much larger point clouds without extreme computational resources or manual pre-processing, just a lot of processing time.
Dead tree detection
Reduce memory requirements for final segmentation step.
Improved segmentation model with expanded datasets.
-----------------------------------------------------------------------------------------

Instructions for training a new semantic segmentation model

FSCT relies heavily on the segmentation model working properly. Training your own model may help expand the utility of FSCT to additional datasets outside of the original training set I used.
Step 1 - Creating training data

Unless you modify the code, training data must be provided as a .las file. This file must have a "label" column, with integer based labels as follows: 1: Terrain, 2: Vegetation, 3: Coarse woody debris, 4: Stems/branches.

Look at a "segmented.las" or "segmented_cleaned.las" file (an output of FSCT in normal use) as an example of what the training data must look like. It is strongly recommended to use FSCT to label your data, THEN correct it manually.

Note: manually segmenting/correcting point clouds is extremely tedious. The original dataset took me ~3-4 weeks to label from scratch... I use CloudCompare's segmentation tool for manually correcting the training data. You should start by loading the terrain_points.las, vegetation_points.las, cwd_points.las, and stem_points.las. I may eventually add an explanation video of how I do this, but for now, you will need to work out a way to do this. Importantly, take great care to label consistently. Sloppy labelling may result in your model not learning what you want it to learn. Small details can matter.
Step 2 - Preparing training data for processing

Take your chosen point cloud, and chop it into train, validation and test slices. You may choose to slice them as 50%, 25% and 25% respectively, but use your discretion.

    Save each slice as a .las file.
    Place the "train" slice into the directory FSCT/data/train_dataset/
    Place the "validation" slice into the directory FSCT/data/validation_dataset/
    Place the "test" slice into the directory FSCT/data/test_dataset/

You can have multiple point clouds in the above directories, and during preprocessing, they will all be placed in the respective sample directories FSCT/data/*_dataset/sample_dir/
Step 3 - Preprocessing the training data

Set the parameters: preprocess_train_datasets, preprocess_validation_datasets and preprocess_test_datasets to True (or 1). Run the train.py file and it will generate the samples for you. After running this the first time, set the above to False (or 0) to avoid preprocessing them again and duplicating them in the sample_dir directories.

For each labelled point cloud you wish to use for training, you must slice it into a chunk for training (most of the point cloud), and a chunk for validation. Place the training chunk into the "data/train_dataset/" directory.

Note: Preprocessing will add files to the respective sample_dir directory, but does not yet delete them. This is important if you re-run the preprocessing step.
Here is a simple scenario which should hopefully make this clearer:

I have already preprocessed some point clouds located in the train_dataset directory. I have created another training dataset and wish to preprocess it so I can use it for training.

I have 2 options: Option A: move the already processed point clouds out of the train_dataset directory. Leave the sample_dir directory as it was. Add the new training point cloud into the train_dataset directory. Set the preprocess_train_datasets parameter to 1 and run the script. As you moved the previously processed point clouds out of the train_dataset directory, they will not be processed, and just the new point cloud will be pre-processed and added to the sample_dir directory. Set the preprocess_train_datasets parameter back to 0 and proceed as you wish.

Option B: Leave your previously processed training point clouds in the train_dataset directory, add your new training point cloud to this directory also. Manually delete the contents of the sample_dir directory and re-run preprocessing for all of the training point clouds.

Options A and B achieve the same thing, but option A is more efficient, as you are not pre-processing everything from scratch again. Option B is likely necessary if you wish to remove a sample point cloud from the dataset.

While most users of FSCT aren't likely to be training their own models, I plan to improve this process. Please see here for future work enhancements planned: #4
Step 4 - Train the model

You can either let the script continue on after the preprocessing step, or stop it, turn off the preprocessing modes and rerun. Be sure to set the parameters according to your computer's specs. If you have CUDA errors, reduce the batch size or switch to CPU mode. If you don't have an Nvidia GPU, you must use CPU mode, but training will be very slow...

The training_monitor.py script will plot the loss and accuracy of the model. You must run this simultaneously in a separate terminal/python console to the training script.

Note: the training process will take several days on a powerful desktop computer.
Step 5 - Use the trained model in FSCT

Simply change the model_filename in other_parameters.py to the model you named in train.py.
An idea potentially worth exploring

FSCT is already capable of producing reasonably well segmented point clouds (within the stated limitations). By leveraging FSCT to automatically segment point clouds, it seems likely that the model could almost train itself into a more consistent and robust state through the use of carefully designed data augmentations.
---------------------------------------------------

Create a plan file and interview me in detail using AskUserQuestionTool about literally anything: technical implementation, UI $ UX, concerns, trade offs, etc.

I forked FSCT years ago and now since the program is no longer being supported by the developer I would like to bring it into the modern age. There needs to be a nice GUI that has access to all of the program functions and settings. There should be a section for creating training data. There needs to be a way to select the models to be used when running a project.
The input data to be processed is Point Clouds. These can be up to 500 million points. We need a good way to visualize the clouds in the app. Tools like the Circular Plot option should be visualized in the viewer so you know what your cropping.
Use tool tips to give good understanding of what each button and tool do. Use hover effects etc that a modern program would use.
All outputs should be configurable for location and content. Plot report should use the Understory  Logo and colors. There should be a way to add notes and project info 
There is a icon and logo in the FSCT root. Use these in the new app and create a color scheme that will match them.
las output is ok. las is what it currently outputs. I would like pcd or las input ability
Write a in depth step by step tutorial for training a new semantic segmentation model.
Make sure to have a guided process in the app to train new model. There needs to be a way to take a untrained unlabeled pcd or las file and use it. There needs to be a way to correct the point cloud segmentation by the user. 
The original project used Conda. I would like to use a venv if we don't need Conda.
GPU's PyTorch Python have all changed since this was created. Move it to updated software. I have a Blackwell RTX 6000 GPU. It and other 5000 series cards need at least PyTorch 2.7.0 with CUDA 12.8.

Tree numbering should line up scan to scan and run to run so data can be compared.
Install full dependencies and environment to run
Write and implement testing of each part that you perform. 

---------------------------------------------------------------
Phase 1: Configuration System & Core Refactoring

 Goal: Replace hardcoded parameter dicts with structured config. Add PCD input support.

 1.1 Configuration dataclasses (understory/config/settings.py)

 Extract parameters from run.py (lines 20-51) and other_parameters.py into:
 - ProcessingConfig — plot_centre, plot_radius, slice_thickness, etc.
 - ModelConfig — model_filename, box_dimensions, min/max_points_per_box
 - OutputConfig — output_directory, content toggles
 - ProjectConfig — top-level, includes project name/notes, saves as YAML

 Bridge method to_legacy_params() converts structured config back to the flat dict all
 existing pipeline classes expect — enabling incremental migration.

 1.2 Refactor shared path construction (understory/core/paths.py)

 All 5 pipeline classes duplicate the same output_dir construction logic. Create FSCTPaths
 class that standardizes this.

 1.3 Add PCD input support (scripts/tools.py)

 Extend load_file() and save_file() to handle .pcd via Open3D. Keep LAS as default output.

 1.4 Pipeline wrapper (understory/core/pipeline.py)

 Modern wrapper around FSCT() from run_tools.py. Accepts ProjectConfig, converts to legacy
 params, runs pipeline. Accepts optional progress_callback for GUI integration.

 1.5 Verification

 - YAML project save/load round-trips
 - Pipeline runs through new wrapper with identical output
 - PCD files load correctly

 ---
 Phase 2: GUI Foundation & Point Cloud Viewer

 Goal: PySide6 app shell with 3D PyVista point cloud viewer capable of handling 500M points
 via LOD.

 2.1 Main window layout (understory/gui/main_window.py)

 +------------------------------------------------------------------+
 |  [Understory Logo]  File  View  Tools  Help                       |
 +------------------------------------------------------------------+
 |  [Sidebar/Tool Panel]   |  [3D Point Cloud Viewer]                |
 |                          |                                         |
 |  Project Settings       |                                         |
 |  Input File(s)          |                                         |
 |  Pipeline Controls      |                                         |
 |  Model Selection        |                                         |
 |  Output Config          |                                         |
 +-------------------------+-----------------------------------------+
 |  [Status Bar: Progress, GPU info, point count]                    |
 +------------------------------------------------------------------+

 2.2 Point cloud viewer (understory/gui/viewer/point_cloud_viewer.py)

 LOD strategy for 500M points:
 - Level 0 (overview): ~1M points, voxel-downsampled
 - Level 1 (medium): ~5M points
 - Level 2 (close): ~20M points, region-of-interest
 - Full resolution stored in memory for processing only
 - Color modes: RGB, height gradient, classification, tree_id

 Interactive tools:
 - Circular plot preview (adjustable center + radius + buffer)
 - Point picking/info display
 - Section/clip planes

 2.3 Branding & theme

 Color palette from icon:
 - #1a4a3a dark forest (backgrounds, headers)
 - #2d7a5e medium forest (primary actions)
 - #4a9e7e medium green (accents, success)
 - #a8d8c0 light mint (highlights, hover)
 - QSS stylesheet applying palette to all widgets

 2.4 Tooltips system (understory/gui/tooltips.py)

 Centralized tooltip text for every parameter, derived from run.py and other_parameters.py
 inline comments.

 2.5 Verification

 - App launches with branding, icon in taskbar
 - Can open LAS/PCD and display in viewer
 - 500M points don't crash (LOD kicks in)
 - Rotate/pan/zoom works smoothly
 - Circular plot preview renders correctly
 - Tooltips on all controls

 ---
 Phase 3: Pipeline Integration with GUI

 Goal: Run the processing pipeline from the GUI with progress, model selection, and results
 display.

 3.1 Progress reporting (understory/core/progress.py)

 Qt Signal-based progress system. Modify each pipeline class to call progress_callback at key
 points (currently all just print()):
 - preprocessing.py:108 — "Pre-processing..."
 - inference.py:115 — batch X/Y progress
 - measure.py:820 — slice height loop
 - report_writer.py:49 — report generation

 3.2 Processing panel (understory/gui/panels/processing_panel.py)

 - File input with drag-and-drop
 - All parameters in collapsible sections with tooltips
 - Pipeline stage checkboxes
 - Run button (executes in QThread)
 - Per-stage + overall progress bar
 - Console log output area

 3.3 Model selection (understory/gui/panels/model_panel.py)

 - Scan /model/ for .pth files
 - Dropdown to select model
 - Import button for new models
 - Modify inference.py to accept model path from config

 3.4 Output configuration (understory/gui/panels/output_panel.py)

 - Output directory picker
 - Checkboxes per output file type (point clouds, CSVs, visualizations, report)

 3.5 Results viewer

 After pipeline completes:
 - Auto-load segmented cloud colored by classification
 - Tree measurements in data table
 - Stem map and histograms displayed in-app
 - CLI mode preserved: python -m understory --cli

 3.6 Verification

 - Full pipeline runs from GUI button
 - Progress bar updates in real-time
 - Results auto-display after completion
 - All output files land in selected directory
 - Headless CLI still works

 ---
 Phase 4: Report Modernization

 Goal: Branded reports with Understory colors, project metadata, configurable content.

 4.1 HTML report template

 Replace plain markdown generation in report_writer.py with Jinja2 HTML template:
 - Understory logo header
 - Brand color scheme
 - Project info fields (name, operator, date, notes)
 - Data tables for tree measurements
 - Embedded stem map and histograms
 - Print-friendly CSS

 4.2 Enhanced visualizations

 - Brand colors in matplotlib plots (#4a9e7e instead of generic green)
 - Understory watermark on stem map
 - Summary statistics table

 4.3 Verification

 - Report renders correctly in browser
 - Branding visible throughout
 - Project metadata present
 - Configurable sections include/exclude works

 ---
 Phase 5: Training Data Pipeline & Label Correction

 Goal: Guided in-app training workflow replacing the current "use CloudCompare manually"
 process.

 5.1 Training panel (understory/gui/panels/training_panel.py)

 Guided workflow:
 1. Import unlabeled point cloud (LAS/PCD)
 2. Run initial segmentation (bootstrap labels with existing model)
 3. Review & correct labels in-app
 4. Export labeled data
 5. Configure & run training

 5.2 Label correction tool (understory/gui/viewer/label_editor.py)

 Extends point cloud viewer with:
 - Selection tools: box select, lasso select, sphere/brush select
 - Class painting: select points → assign class (Terrain=1, Veg=2, CWD=3, Stem=4)
 - Color by class with legend
 - Undo/redo stack
 - Keyboard shortcuts: 1-4 for class assignment

 5.3 Training execution

 - Expose all training params from train.py (epochs, LR, batch size, device)
 - Run training in QThread with live loss/accuracy plots (replaces training_monitor.py)
 - New models auto-appear in model selection

 5.4 Training tutorial (docs/training_tutorial.md)

 Step-by-step guide: data collection → bootstrap → correction → training → evaluation →
 inference.

 5.5 Verification

 - Full label correction round-trip works
 - Training runs from GUI with live metrics
 - Trained model usable for inference

 ---
 Phase 6: Persistent Tree Numbering

 Goal: Consistent tree IDs across scans/runs for longitudinal comparison.

 6.1 Tree registry (understory/core/tree_registry.py)

 - JSON registry file stores {tree_id: {x_base, y_base, dbh, scan_history}}
 - Matching algorithm: KD-tree spatial matching on tree base coordinates within configurable
 radius (default 2m), DBH similarity as tiebreaker
 - New trees get max_id + 1
 - Unmatched old trees preserved in registry

 6.2 Pipeline integration

 After measure.py generates tree_data, before saving CSVs:
 - Load registry → match trees → assign persistent IDs → update registry

 6.3 Registry viewer in GUI

 - Table showing all registered trees with ID, position, DBH history
 - Growth metrics across multiple scans

 6.4 Verification

 - Process same plot twice → IDs match
 - Slightly shifted plot → IDs still match
 - New tree → gets new ID
 - Removed tree → other IDs unaffected

 ---
 Phase 7: Polish, Testing & Packaging

 - Performance profiling on 500M point clouds
 - Fix measure.py O(n^2) np.vstack() calls (use list accumulation)
 - Test suite (unit + integration)
 - Error handling: GPU OOM recovery, input validation, GUI error dialogs
 - Resume from mid-pipeline failures
 - pyproject.toml entry points
 - Desktop shortcut / packaging

 ---
 Phase Dependencies

 Phase 0 (Environment)
     ↓
 Phase 1 (Config + Core)
     ↓
     ├── Phase 2 (GUI) ──→ Phase 3 (Pipeline+GUI) ──→ Phase 5 (Training)
     ├── Phase 4 (Reports)
     └── Phase 6 (Tree IDs)
                     ↓
               Phase 7 (Polish)

 Phases 2, 4, and 6 can run in parallel after Phase 1.

 Key Files to Modify

 ┌─────────────────────────────┬─────────────────────────────────────────────────┐
 │            File             │                     Changes                     │
 ├─────────────────────────────┼─────────────────────────────────────────────────┤
 │ scripts/tools.py            │ Fix get_fsct_path(), add PCD support            │
 ├─────────────────────────────┼─────────────────────────────────────────────────┤
 │ scripts/inference.py        │ Fix torch.load(), accept model path from config │
 ├─────────────────────────────┼─────────────────────────────────────────────────┤
 │ scripts/train.py            │ Fix torch.load(), expose progress               │
 ├─────────────────────────────┼─────────────────────────────────────────────────┤
 │ scripts/measure.py          │ Tree registry integration, progress callbacks   │
 ├─────────────────────────────┼─────────────────────────────────────────────────┤
 │ scripts/report_writer.py    │ Brand colors, template system                   │
 ├─────────────────────────────┼─────────────────────────────────────────────────┤
 │ scripts/other_parameters.py │ Migrate to config dataclasses                   │
 ├─────────────────────────────┼─────────────────────────────────────────────────┤
 │ scripts/run.py              │ New entry point via __main__.py                 │
 ├─────────────────────────────┼─────────────────────────────────────────────────┤
 │ requirements.txt            │ Complete rewrite                                │


