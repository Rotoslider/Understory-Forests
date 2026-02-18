cd ~/projects/FSCT
./venv/bin/python -m understory
------------------------------------------------
Commit fea323f is your safety net.

Main App opens in a windowed view which is great. If I try to drag the window it changes to full screen for a second then back to original size and then I can drag without issue. Only happens first drag after launch.
Issues with the Label Editor:
after selecting or highlighting an area the screen pops back to default full view location. When I select points to move to another label I have to zoom back in to where I was viewing before I made the selection.
selecting or deselecting a class resets to full view. It would be helpful to toggle them on and off without it zooming out. It would also be helpful to be able to switch to a top or side view without it zooming all the way back out.
Last time you tried to fix this you broke the confidence tools and the ability to select points.  make sure your changes do not break some other part of the functionality.


------------------------------------------------------------
Add more labels to classify. canopy plus understory brush. 
Add Dead tree detection
---------------------------------

 It would be helpful when preparing a point cloud to be able to draw a poly-line and or box around an area of cloud to remove or keep points. Typically a scan captures unwanted data at the edges that need trimmed to get the proper plot area.


 
-----------------------------------------------------------------------------------------------------------------
## Additional Functionality to Implement

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

-------------------------------------------------------
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


