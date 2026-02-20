cd ~/projects/Understory_Forests
./venv/bin/python -m understory
------------------------------------------------
A couple of minor this to adjust in the App:
Console Output: The measuring plot time needs updated: Measuring plot took 609.0820257663727 s (Change to Minutes and Seconds)
Allometric Equations Window: CrownVolume result column is not in the Preview sheet Need a Output to CSV for Preview sheet.
There should be a way that when a new formula is added its output has a field in the Preview sheet and is exported with the rest
Growth Dashboard: every other tree in the Select Trees Window is highlighted to dark. IT should be a light color like the one used in the Scan History Window
Dragging the Side bar in the main app window does not work correctly. Its snapping to wrong dimensions. It goes between way to narrow to just right for all tabs except Results. Too narrow for Results tab the buttons are clipped.
Under Prepare tab when cropping outliers and or sub sampling after saving the point cloud it should have option to reload new cropped and or subsampled point cloud as the input cloud used in the project from then on. and update the path in the input file on the Project tab.
Another change is very much needed to clean up a point cloud and make it suitable for use It can be on the prepare tab and saved like the crop outliers and or sub sampled cloud
It would be helpful when preparing a point cloud to be able to draw a poly-line and or box around an area of cloud to remove or keep points. Typically a scan captures unwanted data at the edges that need trimmed to get the proper plot area.

------------------------------------------------------------
Add more labels to classify. canopy plus understory brush. 
Add Dead tree detection

-----------------------------------------------------------------------------------------------------------------
 Implementation Plan: 13 Features (Training, QoL, Performance)
 ---
 5. Training Tutorial
 Files: understory/gui/panels/training_panel.py, understory/gui/tooltips.py
 - Add collapsible "Quick Start Guide" QGroupBox (checkable, collapsed by default) at top of
 training panel
 - Add _add_help_button() helper that inserts a small "?" QPushButton into each step's group
 box
 - Add missing tooltip entries in tooltips.py for model_filename, train_batch_size, etc.

 6. Fix O(n^2) vstack in measure.py
 Files: scripts/measure.py, scripts/train.py
 measure.py (~lines 940-1008, the tree assignment loop):
 - The complication: sorted_full_cyl_array is rebuilt into a cKDTree each iteration, so it's
 not a simple collect-then-vstack
 - Solution: Pre-allocate output array at max possible size (full_cyl_array.shape[0] * 2
 rows), use a write_idx pointer to fill it incrementally. Build cKDTree from
 sorted_full_cyl_array[:write_idx, :3]
 - This eliminates ALL vstack overhead — both accumulation and the per-iteration rebuild
 - Small local vstacks (appending interpolated cylinders to tree) are fine (small arrays)
 train.py (line 327, running_point_cloud_vis):
 - Replace with vis_parts = [] + vis_parts.append(...) + single vstack when saving

 9. Live Training Loss Plots
 Files: scripts/train.py, understory/gui/panels/training_panel.py
 train.py changes:
 - Add progress_callback=None param to TrainModel.__init__()
 - After each epoch's metrics are computed (~line 350), call self._progress_callback(epoch,
 epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc) if set

 training_panel.py changes:
 - Expand signal: progress = Signal(int, float, float, float, float) (epoch, train_loss,
 train_acc, val_loss, val_acc)
 - TrainingWorker.run(): pass progress_callback=self._on_epoch to TrainModel; _on_epoch emits
 signal
 - Add TrainingChartCanvas(FigureCanvasQTAgg) class (follow growth_panel.py pattern)
   - Two lines: train loss (green) + val loss (orange)
   - Forest-themed styling matching existing charts
   - add_epoch() method: append data, clear+redraw
   - reset() method: clear all data
 - Embed chart in Step 5 group box above progress bar
 - _on_train_progress(): update chart + progress bar + status text

 11. Multiprocessing for Preprocessing
 File: scripts/preprocessing.py
 - Replace threading.Thread calls (lines 153-179) with
 multiprocessing.get_context("spawn").Pool
 - threaded_boxes() is already a static method with picklable args — clean fit for
 Pool.starmap()
 - Distribute point_divisions across workers, compute per-worker id_offsets
 - Rename to process_boxes() for clarity
 ---

-----------------------------------------------------------------------------------------------
# TO DO
### Infrastructure
App image for Linux 
| Feature | Description | Complexity |
|---------|-------------|------------|
| **Windows/macOS testing** | Verify installation and GUI on Windows and macOS | Medium |
| **CI/CD pipeline** | GitHub Actions for automated testing on push | Medium |
| **Standalone packaging** | PyInstaller or cx_Freeze executable for distribution | High |


### Infrastructure part 2
| **Plugin system** | Third-party measurement or visualization plugins | High |
| **Documentation site** | Sphinx or MkDocs user guide with screenshots | Medium |

-----------------------------------

# Understory Forests — Feature Implementation Plan (Done)

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
-----------------------------------------------------

# ideas from Sean K

CWD Volume
Individual Tree files in folder with option. Option also for vegetation or no veg with each tree.
Adjust tree id labels to not skip numbers (just part of the process currently as the id labels are arbitrary).
Explore potential optimizations related to global shifting. May be able to use 32 bits in much of the code if returning the global shift happens at the end.
Rectangular, tree-aware-plot-cropping mode capable of automatically processing much larger point clouds without extreme computational resources or manual pre-processing, just a lot of processing time.
Dead tree detection
Reduce memory requirements for final segmentation step.
Improved segmentation model with expanded datasets.


---------------------------------------------------
# My PLAN
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
 Branding & theme
 Color palette from icon:
 - #1a4a3a dark forest (backgrounds, headers)
 - #2d7a5e medium forest (primary actions)
 - #4a9e7e medium green (accents, success)
 - #a8d8c0 light mint (highlights, hover)
 - QSS stylesheet applying palette to all widgets

 
