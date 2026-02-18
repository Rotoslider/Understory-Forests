# Understory Project Status

**Updated:** 2026-02-17

## Current State

All core pipeline features are functional. The GUI provides a complete workflow from point cloud loading through processing, measurement, reporting, and result comparison.

**Test suite:** 76 tests (73 fast, 3 slow E2E) — all passing.

---

## Completed Features

| Feature | Notes |
|---------|-------|
| PySide6 GUI with 5-tab sidebar | Project, Prepare, Process, Advanced, Results |
| 3D point cloud viewer (PyVista) | LOD at 1M/5M/20M, Eye-Dome Lighting, 4 color modes |
| Full pipeline integration | All 5 stages via QThread, weighted progress bar |
| Interactive plot circle | Drag in 3D + spinbox control, synced both ways |
| Point cloud preparation | Axis swap/rotate, crop outliers, voxel subsample, save prepared cloud |
| Dataclass config + YAML | ProjectConfig with `to_legacy_params()` bridge, 30+ tooltips |
| Project management | YAML project files, timestamped run folders, config snapshots |
| Run comparison | Pipeline Run History combo in Results tab, browse any previous run |
| Persistent tree IDs | Tree registry with KD-tree spatial matching across runs |
| Branded HTML reports | Point cloud stats, coverage, tree measurements, stem map, histograms, timing |
| PDF export | One-click A4 PDF via QtWebEngineWidgets |
| Report file organization | Reports saved to runs/reports/, data to runs/output/ |
| GPU live monitoring | nvidia-smi polling during pipeline, status bar display |
| Tree-aware plot cropping | Auto buffer=0.5m when radius set, prevents processing outside circle |
| 19 output layer checkboxes | All pipeline .las outputs selectable and loadable into viewer |
| Tree data export | CSV export defaulting to reports/ folder |
| Tree highlight in viewer | Click row in table to focus camera on that tree |
| Training panel | Dataset selection, hyperparameters, execution (no live plots) |
| Label editor | Box selection for relabeling points |
| Camera views | Top/Front/Right/Isometric presets, click-to-focus |
| PCD format support | Automatic conversion to LAS for pipeline |

---

## Remaining Tests to Write

### GUI Tests (High Priority)

Require `pytest-qt` and a `QApplication` fixture.

| Test | Description |
|------|-------------|
| `test_main_window_init` | MainWindow creates without error, all panels exist |
| `test_processing_panel_build_config` | `_build_config()` produces valid `ProjectConfig` from UI defaults |
| `test_processing_panel_apply_config` | `_apply_config()` round-trips: build, apply, build produces identical config |
| `test_file_loaded_signal` | Setting an input file emits `file_loaded` with correct path |
| `test_pipeline_started_signal` | Starting pipeline emits `pipeline_started` signal |
| `test_pipeline_error_signal` | Pipeline failure emits `pipeline_error` signal |
| `test_run_combo_population` | `_populate_runs()` fills combo from a mock project folder |
| `test_run_selection_refreshes_results` | Selecting a run calls `_populate_results()` with correct output_dir |
| `test_layer_checkboxes_enable_disable` | `_populate_results()` enables only checkboxes for existing .las files |
| `test_report_buttons_enable` | Report/PDF buttons enable when report exists in reports/ or output/ |
| `test_export_pdf_path` | `_export_pdf()` writes PDF to reports/ folder |
| `test_export_tree_data_default_path` | Export tree data dialog defaults to reports/ folder |
| `test_gpu_monitor_signal` | `GpuMonitor._poll()` emits `updated` with formatted string |
| `test_gpu_monitor_lifecycle` | Monitor starts on `pipeline_started`, stops on finish/error |

### Report Tests

| Test | Description |
|------|-------------|
| `test_report_point_cloud_stats` | Template renders point cloud statistics when `num_points_original > 0` |
| `test_report_coverage_section` | Template renders coverage section when any coverage value > 0 |
| `test_report_time_format` | Times render as "X min Y s", total as "X.X min" |
| `test_report_no_trees` | Report renders when `num_trees == 0`, point cloud stats still shown |
| `test_export_pdf_creates_file` | `export_pdf()` creates a valid PDF |
| `test_report_files_in_reports_dir` | After pipeline, report files in reports/ and not in output/ |

### Pipeline Tests

| Test | Description |
|------|-------------|
| `test_buffer_auto_set` | `plot_radius > 0` and `buffer == 0` sets buffer to 0.5 |
| `test_tree_registry_integration` | tree_data.csv has persistent IDs from tree_registry.json |
| `test_tree_registry_consistency` | Two identical runs produce identical TreeIds |
| `test_report_files_moved` | Report PNGs and HTML moved from output/ to reports/ |

### Viewer Tests

| Test | Description |
|------|-------------|
| `test_viewer_load_points` | Viewer loads points, updates point count |
| `test_viewer_color_modes` | All 4 color modes render without error |
| `test_viewer_camera_presets` | Camera preset methods execute without error |
| `test_viewer_plot_circle` | Plot circle shows/hides based on radius |
| `test_viewer_tree_highlight` | Tree ID selection focuses camera on correct points |

### Tree Registry Tests (additions)

| Test | Description |
|------|-------------|
| `test_registry_across_runs` | Two match_trees calls with shuffled order produce same IDs |
| `test_registry_file_persistence` | Registry loads correctly from saved JSON after restart |

---

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
