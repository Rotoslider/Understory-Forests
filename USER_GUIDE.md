# User Guide — How to Use Understory to Measure Your Forest

This guide walks you through every part of the Understory program. It covers how to set up a project, prepare your point cloud, run the pipeline, understand the advanced settings, and view your results.

**You do not need programming experience to follow this guide.** Just follow each section in order for your first time. After that, use it as a reference whenever you need to look something up.

---

## What Does Understory Do?

Understory takes a 3D scan of your forest (called a "point cloud") and automatically:

1. **Identifies** what each point is — ground, tree trunk, leaf, or fallen wood
2. **Finds** individual trees in the scan
3. **Measures** each tree — diameter, height, and volume
4. **Creates a report** with all measurements, maps, and charts

This saves you from having to measure every tree by hand in the field.

---

## Starting the Program

Open a terminal and type:

```
./venv/bin/python -m understory
```

The program opens with:
- A **sidebar** on the left with five tabs: Project, Prepare, Process, Advanced, Results
- A **3D viewer** on the right where you see your point cloud
- A **status bar** at the bottom showing progress and GPU information

**Tip:** You can also open files by dragging them from your file manager and dropping them directly onto the Understory window. Point cloud files (.las, .laz, .pcd) are loaded into the viewer. Project files (.yaml) are opened as projects.

---

## The Project Tab

This is where every project begins.

### Opening Your Point Cloud

1. Click **Browse** next to "Input File"
2. Find your scan file (.las, .laz, or .pcd) and select it
3. The point cloud loads in the 3D viewer on the right

### Project Information

Fill in these fields:

- **Project** — A name for this project. Use something that helps you find it later.

  Good names:
  - `Rio Negro Plot 1 — February 2026`
  - `Mangrove Site A — Dry Season`
  - `Cerrado Transect 3`

  Bad names:
  - `test`
  - `scan1`
  - `data`

- **Operator** — Your name or the name of the person running the scan. This appears in the report so you know who processed the data.

- **Notes** — Anything you want to remember about this scan. Some examples:
  - "Dense understory, many palm trees"
  - "Scanned after heavy rain, some noise expected"
  - "Second scan of this plot, trees 5 and 12 fell since last visit"

  Notes appear in the final report and are saved with the project.

### Output Folder

By default, the program creates the project folder next to your input file. You can change this by clicking the **...** button if you want to save results somewhere else.

### Saving and Opening Projects

- **Ctrl+S** — Save the current project (creates a `project.yaml` file)
- **Ctrl+Shift+S** — Save the project to a new location
- **Ctrl+Shift+O** — Open an existing project

### Recent Projects

The program remembers the last 10 files and projects you opened. Find them under **File > Recent Projects** in the menu bar. This saves you from browsing for the same file again.

### Attaching Field Photos

You can attach photos from the field to your project. Click **Attach Photos** in the Project tab and select one or more image files. The photos are:

- Copied into the project folder so they stay with your data
- Shown as thumbnails in the Project tab
- Included in the HTML report under a "Field Photos" section

This is useful for documenting site conditions, marking specific trees, or recording any visual observations.

### Project Folder Structure

When you save a project and run the pipeline, the program creates this folder structure:

```
Rio Negro Plot 1/
  project.yaml              ← Your project settings
  tree_registry.json         ← Keeps tree IDs consistent between runs
  runs/
    run_2026-02-15_14-30-00/ ← First pipeline run
      run_config.yaml        ← Settings used for this run
      output/                ← All the point cloud and CSV files
      reports/               ← HTML report, PDF, charts
    run_2026-02-16_09-00-00/ ← Second pipeline run
      ...                    ← Different settings or updated data
```

Each time you run the pipeline, it creates a new timestamped folder. This means you never lose previous results — you can always go back and compare.

---

## The Prepare Tab

Before running the pipeline, you may need to clean up your point cloud. This tab has tools for that.

### Orientation — Fixing Which Way Is Up

Some scanners save the data with a different "up" direction. If your trees are growing sideways in the viewer, you need to fix the orientation.

- **Swap Y↔Z** — Most common fix. Use this if trees are growing sideways
- **Swap X↔Z** — Try this if Y↔Z didn't work
- **Swap X↔Y** — Rotates the cloud horizontally (rarely needed)
- **Rotate 90° Z** — Turns the cloud 90 degrees if it's facing the wrong direction
- **Reset Orientation** — Undo all orientation changes and go back to the original

**How to tell if orientation is wrong:** If your point cloud looks like a flat pancake from the front, or trees are horizontal, you need to swap axes. Trees should grow straight up (along the Z axis).

### Clean — Removing Outliers

Click **Crop Outliers** to remove noise points that are far away from the main cloud. These are often caused by:
- Birds or insects passing through the scanner beam
- Reflections from water or shiny surfaces
- GPS errors in the scanner

The program removes points beyond the 99.5th percentile on each axis. This is safe — it only removes the most extreme points.

Click **Reset Crop** if you want to undo the crop and bring all points back.

### Subsample — Reducing Point Density

If your scan has many millions of points, subsampling makes processing faster without losing important detail.

- **Enable subsampling** — Check this to turn on subsampling
- **Min spacing** — The minimum distance between points after subsampling

**Suggested spacing values:**

| Scanner / Data Type | Suggested Spacing | Why |
|---------------------|-------------------|-----|
| High-res TLS (10M+ points) | 0.01 m (1 cm) | Keeps fine detail while reducing count |
| Medium-res TLS (1-10M points) | 0.005 m (5 mm) | Light reduction, preserves most detail |
| Photogrammetry | 0.02 m (2 cm) | Often already noisy, more aggressive is OK |
| Low-res or small clouds (<1M) | Don't subsample | Already sparse enough |

Click **Preview Subsample** to see what it looks like before committing. The viewer shows you the subsampled result so you can judge if the spacing is too aggressive.

### Save Prepared Cloud

After fixing orientation, cropping outliers, and/or subsampling, click **Save Point Cloud As** to save the cleaned version. This is useful because:
- You can reload the cleaned file next time without repeating these steps
- The pipeline will use the prepared cloud instead of the original
- You have a backup of your cleaned data

### Undo and Redo

If you make a mistake while preparing your point cloud — for example, you swap the wrong axis or crop too aggressively — you can undo the last operation:

- **Ctrl+Z** — Undo the last preparation step
- **Ctrl+Shift+Z** — Redo (bring it back)

The program keeps up to 5 undo steps. This works for axis swaps, outlier cropping, and subsampling.

---

## The Process Tab

This is where you configure and run the pipeline.

### Plot Geometry

#### Plot Radius

The plot radius defines a circular area to process. Only points inside this circle are included in the final results.

- **Set to 0** — Process the entire point cloud (no cropping)
- **Set to a value** — Process only points within this many meters of the plot center

**When to use a plot radius:**
- Your scan covers a larger area than your actual plot
- You have a defined circular plot (common in forestry: 10m, 15m, 20m radius)
- You want to compare plots of the same size

**When to leave at 0:**
- You want to process everything in the scan
- Your scan already covers only the area you need
- You don't have formal plot boundaries

The program draws a **pink circle** in the 3D viewer showing the plot boundary. You can **drag this circle** to reposition it. Use your scroll wheel on the center point to change the drag point size.

#### Plot Buffer

When using a plot radius, the buffer adds extra space around the edge. This prevents cutting trees in half at the boundary. Trees whose base is inside the plot are fully included, even if their canopy extends past the edge.

- **Default: 0 m** — Uses a small automatic buffer
- **Suggested: 2-5 m** — For forests with large canopies

#### Plot Center

- **Auto (checked)** — The program calculates the center from the point cloud bounding box. This is correct for most scans.
- **Manual** — Uncheck "Auto" and type in X, Y coordinates. Use this if your plot center is not the center of the scan.

### Model

The model is the "brain" that identifies what each point is (ground, vegetation, etc.).

- The dropdown shows all models in the `model/` folder
- **model.pth** is the default model that comes with the program
- If you have trained your own model, click **Import Model** to copy it into the model folder

### Performance

#### Batch Size

This controls how many point cloud boxes the GPU processes at the same time during segmentation.

- **Default: 2** — Safe for most GPUs
- **Higher (4-20)** — Faster, but uses more GPU memory
- **Lower (1)** — Currently may not work

**How to choose:** Start with the default. If the pipeline finishes without errors and you want it faster, try increasing to 4, then 8, then 16. If you get a memory error, go back down.

| GPU Memory | Suggested Batch Size |
|-----------|---------------------|
| 4 GB | 1-2 |
| 8 GB | 2-4 |
| 16 GB | 4-8 |
| 24+ GB | 8-20 |
| 95 GB | 20+ |

#### CPU Cores

Controls how many CPU cores are used for parallel processing (preprocessing and measurement stages).

- **0** — Use all available cores (fastest, recommended)
- **1-4** — Use fewer cores if your computer struggles or you need to do other work while processing

#### CPU Only

Check this if you do not have an NVIDIA GPU. The pipeline will run on CPU only. It is much slower but works on any computer.

### Pipeline Stages

You can turn individual stages on or off. Each checkbox controls one stage:

| Stage | What It Does | When to Disable |
|-------|-------------|-----------------|
| **Preprocessing** | Cuts the cloud into boxes for the model | Almost never — needed for segmentation |
| **Semantic Segmentation** | Labels each point (terrain, vegetation, CWD, stem) | If you already ran segmentation and only want to re-measure |
| **Post-processing** | Builds the DTM, cleans labels, separates point classes | Rarely — needed for measurement |
| **Measurement** | Finds trees, measures DBH, height, volume | If you only need segmented point clouds, not measurements |
| **Generate Report** | Creates HTML report with charts and maps | If you don't need a report |

**Most common usage:** Leave all five checked for a complete run.

**Re-running just measurement:** If you already ran the full pipeline but want to try different measurement settings (like different CCI or height percentile), you can uncheck Preprocessing, Segmentation, and Post-processing, then run again. This is much faster because segmentation (the slowest step using CPU only) is skipped.

---

## The Advanced Tab

These settings control the detailed behavior of the measurement and segmentation stages. **The defaults work well for most forests.** Only change them if you understand what they do and have a reason to adjust them.

**Right-click any setting to reset it to its default value.** The menu shows you what the default is.

### Measurement Settings

#### DTM Resolution (default: 0.5 m)

The Digital Terrain Model is a grid that represents the ground surface. This value is the size of each grid cell.

- **0.5 m** — Good for most forests
- **Smaller (0.2-0.3 m)** — More detailed ground model, useful for steep or uneven terrain. Takes longer to compute.
- **Larger (1.0+ m)** — Smoother ground model, faster. Use for flat terrain or very large areas.

**When to change:** If the ground in your forest is very uneven (many rocks, steep slopes, root mounds), try 0.3 m for a more accurate ground model. On flat terrain, 0.5 or even 1.0 m is fine.

#### Min CCI (default: 0.3)

CCI stands for "Circumferential Completeness Index" — how much of the circle around a tree trunk has data points. 1.0 means the scanner captured the entire circle. 0.5 means only half the circle has data (common with a single scanner position).

Cylinder measurements with a CCI below this threshold are thrown out.

- **0.3** — Good general default. Keeps most measurements but removes very poor ones.
- **Lower (0.1-0.2)** — Keeps more measurements, even low-quality ones. Use if you have sparse data.
- **Higher (0.4-0.5)** — Stricter quality filter. Use if you want only high-confidence measurements.

**When to change:** If you scanned from only one position (single-scan TLS), the maximum CCI is about 0.5 because the scanner can only see one side of each tree. Keep Min CCI well below 0.5 (like 0.2-0.3) to avoid losing all your measurements.

#### Min Tree Cylinders (default: 10)

Minimum number of cylinder measurements required to count something as a tree. Trees with fewer cylinders are deleted from the results.

- **10** — Good default. Filters out small noise clusters.
- **Lower (3-5)** — Keeps smaller trees that only have a few measurements. Use for young forests or sparse data.
- **Higher (15-20)** — Stricter filter. Use if you're getting false trees (noise being counted as a tree).

#### Min Cluster Size (default: 30)

The program uses a clustering algorithm (HDBSCAN) to group stem points into individual trees. This is the minimum number of points required to form a cluster.

- **30** — Good default for most data
- **Lower (10-20)** — Detects smaller trees but may create false clusters from noise
- **Higher (50-100)** — Only detects trees with many stem points. Use for dense, high-quality scans.

**When to change:** If small trees are being missed, try lowering this to 15-20. If you're getting false detection's (noise counted as trees), increase it.

#### Height Percentile (default: 100)

The percentile used for calculating maximum tree height. 100 means the actual highest point is used.

- **100** — Use for clean data with no noise above the canopy
- **98-99** — Use if your data has noise points above the trees (common with photogrammetry or some TLS setups). This ignores the top 1-2% of points, which are likely noise.

**When to change:** If tree heights look unrealistically tall in your results (e.g., 80 m trees in a forest where the tallest should be 40 m), there is probably noise above the canopy. Set this to 98 or 99.

#### Tree Base Cutoff (default: 5.0 m)

The maximum height above ground where a valid tree base can be found. If a tree has no cylinder measurements below this height, it is removed.

- **5.0 m** — Good for most forests
- **Higher (8-10 m)** — Use if scanning very tall trees with high buttress roots or if the scanner couldn't capture the lower trunk
- **Lower (2-3 m)** — Stricter filter. Use if you're getting false trees detected in the canopy.

**When to change:** In tropical forests with buttress roots, the lowest good measurement might be at 3-4 m. Keep this at 5 m or higher for those forests.

#### Ground Veg Cutoff (default: 3.0 m)

Vegetation below this height is classified as "understory" or "ground vegetation" and is not assigned to individual trees.

- **3.0 m** — Good default
- **Lower (1-2 m)** — In open forests with short understory
- **Higher (5-8 m)** — In dense tropical forests where understory vegetation reaches higher

**When to change:** If you notice tall shrubs or small trees being counted as understory rather than individual trees, lower this value.

#### Veg Sorting Range (default: 1.5 m)

Maximum horizontal distance for assigning vegetation points to a tree. Vegetation points further than this from any measured tree cylinder are left unassigned.

- **1.5 m** — Good for typical forests
- **Larger (2-3 m)** — Use for trees with wide canopies
- **Smaller (0.5-1.0 m)** — Use if trees are very close together and you want tighter assignment

#### Stem Sorting Range (default: 1.0 m)

Maximum 3D distance for assigning stem points to a tree. Similar to veg sorting range but for stem/branch points.

- **1.0 m** — Good default
- **Larger (1.5-2.0 m)** — Use for large trees with spreading branches
- **Smaller (0.5 m)** — Use for small, densely packed trees

### Slicing Settings

These control how the program slices the point cloud horizontally to measure tree diameters at different heights.

#### Slice Thickness (default: 0.15 m)

Height of each horizontal slice. The program fits a circle (cylinder) to the points in each slice.

- **0.15 m** — Good for high-resolution data
- **0.2-0.3 m** — Use for lower-resolution data (fewer points per meter of trunk)
- **0.1 m** — Use for very dense data

**When to change:** If cylinder fitting is failing (few or poor measurements), try increasing to 0.2 m. This captures more points per slice, making it easier to fit a circle.

#### Slice Increment (default: 0.05 m)

Vertical distance between the start of each slice. Smaller values mean more slices (more measurements per tree) but slower processing.

- **0.05 m** — Good default, gives detailed measurements
- **0.1 m** — Faster, still good quality
- **0.02 m** — Very detailed but slow. Use only if you need very fine taper measurements.

### Taper Measurement Settings

Taper measurements tell you how the tree diameter changes with height — important for estimating timber volume.

#### Min Height (default: 0.0 m)

Lowest height for taper measurements. Usually 0 (start at ground level).

#### Max Height (default: 30.0 m)

Highest height for taper measurements. Set this to match the tallest trees in your area.

- **30 m** — Good for most temperate and tropical forests
- **50-60 m** — For tall tropical forests with emergent trees
- **15-20 m** — For shorter forests or plantations

#### Height Increment (default: 0.2 m)

Step between taper measurements. A measurement is taken every 0.2 m up the trunk.

- **0.2 m** — Good default
- **0.5 m** — Faster, still useful for basic volume estimates
- **0.1 m** — Very detailed taper profile

#### Taper Slice Thickness (default: 0.4 m)

How thick of a slice to use when finding the diameter at each taper height. Cylinders within +/- half this thickness are considered. The largest diameter at each height is used.

- **0.4 m** — Good default
- **0.6-0.8 m** — For sparse data (captures more points per measurement)

### Segmentation Box Settings

These control how the point cloud is divided into boxes for the deep learning model. **You almost never need to change these.**

#### Box Size (default: 6.0 m)

Size of each box in meters. The point cloud is divided into overlapping cubes of this size.

- **6 m** — Tested and proven default. Leave it alone unless you have a specific reason.
- **Larger (8-10 m)** — More context per box but uses more GPU memory
- **Smaller (4 m)** — Less GPU memory but may miss large trees

#### Box Overlap (default: 0.5)

How much each box overlaps with its neighbors (0.5 = 50% overlap). Higher overlap improves accuracy because each point is seen in multiple boxes.

- **0.5** — Good default
- **Higher (0.6-0.7)** — Better accuracy but much slower
- **Lower (0.3)** — Faster but may miss details at box edges

#### Min Points Per Box (default: 1000)

Boxes with fewer points than this are skipped. This filters out nearly empty boxes.

- **1000** — Good default
- **Lower (500)** — Keep more boxes in sparse areas
- **Higher (2000)** — Skip sparser boxes for faster processing

#### Max Points Per Box (default: 20000)

Boxes with more points than this are randomly subsampled down. This keeps GPU memory usage predictable.

- **20000** — Good default
- **Higher (30000-50000)** — More points per box (may need to lower batch size)
- **Lower (10000)** — Faster but less detail per box

### Cleanup Options

#### Delete Working Directory (default: checked)

The pipeline creates temporary files during processing. This deletes them when done to save disk space.

- **Keep checked** for normal use
- **Uncheck** if you want to re-run just the segmentation or measurement stage without repeating preprocessing. The temporary files let you skip the preprocessing step.

#### Minimize Output Size (default: unchecked)

Deletes non-essential output files, keeping only the tree data CSVs and the report. This saves a lot of disk space for large point clouds.

- **Leave unchecked** if you want to view output layers in the 3D viewer
- **Check** if you only need the final measurements and report, and storage is limited

---

## Running the Pipeline

1. Make sure you have set up the Project tab (input file, project name)
2. Configure the Process tab (plot radius, model, batch size)
3. Click **Run Pipeline** at the bottom of the sidebar (or press **F5**)

### During Processing

- The **dual progress bar** shows two things at once: a thin dark bar on top for overall pipeline progress, and a larger bar below for the current stage. You can see the stage name and overall percentage at a glance
- During semantic segmentation, the progress bar updates batch-by-batch so you can track inference progress
- The **console log** at the bottom shows detailed messages from each stage
- The **status bar** shows GPU memory and utilization in real time
- You can click **Stop** to safely cancel the pipeline (or press **Shift+F5**). The program finishes the current operation cleanly rather than force-stopping, so no data is corrupted

### Processing Times

Processing time depends on point cloud size and your hardware:

| Point Cloud Size | Typical Time (with GPU) |
|-----------------|------------------------|
| < 1 million points | 2-5 minutes |
| 1-5 million points | 5-15 minutes |
| 5-20 million points | 15-45 minutes |
| 20+ million points | 1-3 hours |

The semantic segmentation stage uses the GPU and is usually the fastest part. The measurement stage uses only the CPU and is often the longest part for large clouds.

### After Processing

When the pipeline finishes:
- A success message appears
- The Results tab is ready with your data
- All output files are saved in the run folder
- GPU memory is automatically released

---

## The Results Tab

This is where you view everything the pipeline produced.

### Pipeline Run History

The dropdown at the top shows all previous runs for this project, listed by date and time. Select a run to view its results.

When you select a run, the program also **restores all the settings** that were used for that run. This is very helpful for:
- Remembering what parameters you used
- Re-running with slightly different settings
- Comparing what changed between two runs

### Output Layers

These are the point cloud files the pipeline created. Check the ones you want to see, then click **Load Selected Layers** to display them in the 3D viewer.

**Most useful layers for quick review:**

| Layer | What It Shows | When to Use |
|-------|--------------|-------------|
| **Segmented** | Full cloud with labels + confidence | Check overall segmentation quality |
| **Terrain Points** | Just the ground | Verify ground detection |
| **Stem Points Sorted** | Tree trunks colored by tree ID | See which points belong to which tree |
| **Cleaned Cylinders** | Fitted cylinder models | Check measurement quality |
| **Tree-Aware Crop** | Final cropped output | See what's included in results |
| **DTM** | Ground surface model | Verify the terrain model looks right |

**Tip:** You do not need to load all 19 layers. Start with "Segmented" to check overall quality, then load specific layers if you need to investigate something.

### Color Modes in the Viewer

When viewing output layers, use the color mode buttons in the viewer toolbar:
- **RGB** — Original scan colors
- **Height** — Color by elevation (blue = low, red = high)
- **Classification** — Color by label (brown = terrain, green = vegetation, gold = CWD, red = stem)
- **Tree ID** — Each tree gets a different color

### Tree Measurements Table

Below the output layers, you see a table with measurements for every detected tree:

| Column | What It Means |
|--------|--------------|
| **TreeId** | A number for each tree (stays the same across runs) |
| **DBH** | Diameter at breast height — the trunk diameter at 1.3 m above ground, in meters |
| **Height** | Total tree height in meters |
| **Volume_1** | Volume calculated by adding up all fitted cylinders |
| **Volume_2** | Volume calculated using a cone + cylinder formula |
| **CCI_at_BH** | How complete the measurement is at breast height (0 to 1) |
| **x/y/z_tree_base** | Coordinates of the tree base |

**Click on a row** to highlight that tree in the 3D viewer and fly the camera to it. This is very useful for checking individual tree measurements.

### Persistent Tree IDs

The program remembers trees between runs. If you scan the same plot again next year and run the pipeline, tree #5 will still be tree #5 (as long as it hasn't moved more than 2 meters). This lets you track growth over time.

### Report

- **Open Report** — Opens the HTML report in your web browser. It includes summary statistics, tree measurements, a stem map, distribution charts, and processing times.
- **Export PDF** — Saves the report as a PDF file. Great for printing or sharing.

### Exporting Data

- **Export Tree Data** — Saves the tree measurements table as a CSV file. You can open this in Excel, Google Sheets, or any spreadsheet program.

The CSV file is saved to the run's `reports/` folder by default.

---

## Moving Around the 3D Viewer

| Action | How To |
|--------|--------|
| **Rotate** | Left-click and drag |
| **Pan** (move sideways) | Middle-click and drag |
| **Zoom** | Scroll wheel |
| **Focus on a point** | Press **F**, then right-click a point |
| **Reset view** | Press **Home** |
| **Top view** | Ctrl+1 |
| **Front view** | Ctrl+2 |
| **Right view** | Ctrl+3 |
| **Isometric view** | Ctrl+4 |

**Eye-Dome Lighting (EDL):** Turn this on in the toolbar for better depth perception. It makes shapes easier to see in the point cloud. Highly recommended when checking results.

### Screenshot Export

To save what you see in the 3D viewer as an image:

1. Set up the view you want (rotate, zoom, pick a color mode)
2. Go to **View > Export Screenshot** (or press **Ctrl+Shift+E**)
3. Choose where to save it and pick a format (PNG, JPEG, or TIFF)

The image is saved at 2x resolution for sharp, high-quality output.

### Classification Legend

When you switch to **Classification** color mode, a legend appears in the viewer showing what each color means:

| Color | Class |
|-------|-------|
| Brown | Terrain |
| Green | Vegetation |
| Gold | CWD (Coarse Woody Debris) |
| Red | Stem |

This makes it easy to read the display without memorizing the color scheme.

### Cross-Section Slice

The cross-section tool lets you "slice" through the point cloud to see what's inside — useful for inspecting trunk structure, checking terrain detection, or looking at individual tree layers.

To use it:

1. Find the slice controls in the viewer toolbar
2. Set the **Mode** to Horizontal (cuts by height) or Vertical (cuts by position)
3. Adjust the **Position** slider to move the slice through the cloud
4. Change the **Thickness** to show a thicker or thinner slice

This is purely visual — it hides points temporarily but does not delete or modify anything. Set Mode back to "Off" to see the full cloud again.

### Measurement Tools

You can measure distances directly in the 3D viewer:

1. Go to **Tools > Measure Distance** (for 3D straight-line distance) or **Tools > Measure Height** (for vertical height difference)
2. Right Click a point in the cloud — this sets the first endpoint
3. Right Click a second point — a line appears with the measurement value

**Tips:**
- Press **Escape** to cancel a measurement in progress
- Go to **Tools > Clear Measurements** to remove all measurement lines from the view
- Multiple measurements can be active at the same time

---

## Common Workflows

### First Time Processing a New Site

1. **Project tab** — Open your point cloud, name the project, add notes about the site
2. **Prepare tab** — Crop outliers, subsample if needed, save the prepared cloud
3. **Process tab** — Set plot radius, leave all stages checked, click Run Pipeline
4. **Results tab** — Check segmented output, review tree measurements, export report

### Re-running With Different Settings

1. **Results tab** — Select a previous run to load its settings
2. **Process/Advanced tab** — Change the setting you want to test
3. **Process tab** — Uncheck stages you don't need to repeat (skip preprocessing and segmentation if only changing measurement settings)
4. Click **Run Pipeline**
5. **Results tab** — Select the new run and compare with the old one

### Processing Multiple Plots

For each plot:
1. Create a new project (Ctrl+N)
2. Open the point cloud for that plot
3. Run the full pipeline
4. Export tree data and report

Each project keeps its own folder with all results organized by run.

---

## Analysis Tools

Understory includes several analysis tools for working with your results beyond the basic pipeline. Find them in the **Tools** menu.

### Batch Processing

If you have several point cloud files to process with the same settings:

1. Go to **Tools > Batch Processing**
2. Click **Add Files** to select multiple point cloud files
3. Click **Run** — each file is processed using your current pipeline settings
4. Progress is shown per-file with status icons

The program cleans up GPU memory between files automatically. Each file gets its own run folder.

### Comparing Runs

To compare results from two different pipeline runs (for example, before and after changing settings, or scans from different dates):

1. Go to **Tools > Compare Runs**
2. Select the two run folders to compare
3. Click **Compare** — a comparison report is generated

The report shows:
- Which trees changed in DBH, height, or volume
- Which trees are new (detected in one run but not the other)
- Delta histograms and summary statistics

### Growth Dashboard

If you scan the same plot multiple times (for example, once per year), you can track how individual trees grow:

1. Go to **Tools > Growth Dashboard**
2. Select one or more trees from the list
3. Charts show DBH and height over time for each selected tree

You can export the growth data as a CSV file for further analysis in a spreadsheet.

### Allometric Equations

Allometric equations let you estimate things like above-ground biomass (AGB) and carbon content from your tree measurements:

1. Go to **Tools > Allometric Equations**
2. The program comes with default equations for generic AGB and carbon
3. You can add your own equations using column names from the tree data (like DBH and Height)
4. Results are calculated for all trees and shown in a table
5. Export results as CSV

This is useful for carbon accounting, forest inventory reporting, and research.

### GIS Export

To use your tree data in GIS software (like QGIS or ArcGIS):

1. Go to the **Results tab** and click **Export to GIS**
2. Choose a format:
   - **GeoJSON** — Works everywhere, no extra software needed
   - **Shapefile** — Traditional GIS format (requires the geopandas Python package)
3. Enter a CRS (Coordinate Reference System) string if your data is georeferenced
4. Each tree is saved as a point with all its measurements as attributes

### Point Cloud Comparison

To compare two point clouds visually (for example, two scans of the same plot from different dates):

1. Go to **Tools > Compare Point Clouds**
2. Select a second point cloud file
3. Points are colored by their distance to the nearest point in the other cloud
   - Blue = close (little change)
   - Red = far (big change)
4. Statistics (mean, max, standard deviation) are shown

This is a quick visual check for structural changes between scans.

### Flythrough Animation

Create a smooth camera animation through your point cloud:

1. Go to **View > Flythrough Editor**
2. Position the camera where you want the animation to start
3. Click **Add Keyframe** to capture that position
4. Move the camera to the next position and add another keyframe
5. Repeat for as many keyframes as you want
6. Click **Render** to generate the animation

Export options:
- **Image sequence** — One image per frame (always available)
- **GIF** — Animated image
- **MP4** — Video file (requires the imageio-ffmpeg package)

The camera follows a smooth curved path between keyframes using spline interpolation.

---

## Tips and Best Practices

### Before Scanning
- Place the scanner where you can see as many tree trunks as possible
- Multiple scanner positions around the plot improve CCI (more complete measurements)
- Remove loose vegetation and gear from the scanner's line of sight

### For Best Results
- Always crop outliers before processing — noise points can confuse the model
- Start with default settings — they work for most forests
- If results look wrong, check the segmentation first (load the "Segmented" layer)
- Use the tree measurement table to spot obvious errors (trees with unrealistic DBH or height)

### Saving Storage Space
- Check "Delete working directory" in Advanced (default is on)
- Check "Minimize output size" if you only need measurements, not output point clouds
- Old runs can be deleted manually from the project's `runs/` folder if you don't need them anymore

### Right-Click to Reset

If you changed a setting and can't remember what the original value was, **right-click** on the setting. A menu appears showing "Reset to Default" with the default value. This works for every setting in the Process and Advanced tabs.

---

## Quick Reference

| Task | Where to Find It |
|------|-----------------|
| Open a point cloud | Project tab → Browse |
| Save the project | Ctrl+S |
| Open an existing project | Ctrl+Shift+O |
| Fix orientation | Prepare tab → Swap buttons |
| Remove outliers | Prepare tab → Crop Outliers |
| Subsample | Prepare tab → Enable subsampling → Preview |
| Set plot radius | Process tab → Plot Geometry |
| Choose a model | Process tab → Model dropdown |
| Set batch size | Process tab → Performance |
| Run the pipeline | Run Pipeline button (or F5) |
| Stop the pipeline | Stop button (or Shift+F5) |
| View output layers | Results tab → check layers → Load Selected |
| See tree measurements | Results tab → Tree Measurements table |
| Highlight a tree | Click a row in the tree table |
| Open the report | Results tab → Open Report |
| Export PDF | Results tab → Export PDF |
| Export tree data CSV | Results tab → Export Tree Data |
| Reset a setting | Right-click the setting → Reset to Default |
| Reset camera view | Home key |
| Change colour mode | Viewer toolbar → RGB / Height / Class / TreeID |
| Take a screenshot | View → Export Screenshot (or Ctrl+Shift+E) |
| Undo preparation step | Ctrl+Z (in Prepare tab) |
| Measure distance | Tools → Measure Distance → click two points |
| Compare point clouds | Tools → Compare Point Clouds |
| Batch process files | Tools → Batch Processing |
| Compare two runs | Tools → Compare Runs |
| Track tree growth | Tools → Growth Dashboard |
| Calculate biomass | Tools → Allometric Equations |
| Export for GIS | Results tab → Export to GIS |
| Attach field photos | Project tab → Attach Photos |
| Create flythrough | View → Flythrough Editor |

---

## Keyboard Shortcuts

| Shortcut | What It Does |
|----------|-------------|
| Ctrl+N | New project |
| Ctrl+O | Open a point cloud file |
| Ctrl+Shift+O | Open a saved project |
| Ctrl+S | Save project |
| Ctrl+Shift+S | Save project to new location |
| Ctrl+W | Close the current point cloud |
| F5 | Run the pipeline |
| Shift+F5 | Stop the pipeline |
| Home | Reset the camera view |
| Ctrl+1 | Top-down view |
| Ctrl+2 | Front view |
| Ctrl+3 | Right side view |
| Ctrl+4 | Isometric (3D angle) view |
| Ctrl+Q | Close the program |
| Ctrl+Shift+E | Save a screenshot of the 3D view |
| Ctrl+Z | Undo last preparation step (Prepare tab) |
| Ctrl+Shift+Z | Redo last preparation step (Prepare tab) |
| Escape | Cancel a measurement in progress |
