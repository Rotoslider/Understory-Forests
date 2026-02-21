"""Centralized tooltip text for all Understory parameters.

Derived from inline comments in run.py (lines 20-51) and other_parameters.py.
"""

TOOLTIPS = {
    # Processing parameters (from run.py)
    "plot_centre": (
        "X, Y coordinates of the plot centre in metres.\n"
        "If left empty, the centre is computed from the point cloud bounding box."
    ),
    "plot_radius": (
        "Cylindrical crop radius in metres.\n"
        "Set to 0 to process the entire point cloud without cropping."
    ),
    "plot_radius_buffer": (
        "Extra buffer radius for Tree Aware Plot Cropping Mode.\n"
        "Trees within plot_radius are kept; trees within plot_radius + buffer\n"
        "are used for context but excluded from final results."
    ),
    "batch_size": (
        "Number of point cloud boxes processed simultaneously.\n"
        "Lower this if you get CUDA out-of-memory errors.\n"
        "Must be >= 2. Not relevant for CPU-only mode."
    ),
    "num_cpu_cores": (
        "Number of CPU cores for parallel processing.\n"
        "Set to 0 to use all available cores.\n"
        "Lower this if you run out of RAM."
    ),
    "use_CPU_only": (
        "Run inference on CPU instead of GPU.\n"
        "Use this if you don't have an NVIDIA GPU or lack sufficient VRAM."
    ),
    "slice_thickness": (
        "Height of each horizontal slice for stem measurement (metres).\n"
        "Try 0.2 for lower-resolution point clouds.\n"
        "Very dense clouds may work with 0.1."
    ),
    "slice_increment": (
        "Vertical step between successive slices (metres).\n"
        "Smaller values give better results but increase processing time."
    ),
    "sort_stems": (
        "Sort stem points by tree ID.\n"
        "Required for tree height measurement.\n"
        "Disabling speeds up processing if you only need DBH."
    ),
    "height_percentile": (
        "Percentile for maximum tree height calculation.\n"
        "Set to 98 if your data has noise above the canopy.\n"
        "Leave at 100 for clean data."
    ),
    "tree_base_cutoff_height": (
        "Maximum height above DTM for a valid tree base (metres).\n"
        "Cylinder measurements must exist below this height\n"
        "for a tree to be kept. Filters unsorted branches."
    ),
    "generate_output_point_cloud": (
        "Generate a semantic and instance segmented output point cloud.\n"
        "Overrides the sort_stems setting when enabled.\n"
        "Uses tree-aware plot cropping if configured."
    ),
    "ground_veg_cutoff_height": (
        "Maximum height above DTM for understory vegetation (metres).\n"
        "Vegetation below this height is not assigned to individual trees."
    ),
    "veg_sorting_range": (
        "Maximum horizontal distance to match vegetation to a tree (metres).\n"
        "Vegetation further than this from any cylinder is unassigned."
    ),
    "stem_sorting_range": (
        "Maximum 3D distance to match stem points to a tree (metres).\n"
        "Stem points further than this from any cylinder are unassigned."
    ),
    "delete_working_directory": (
        "Delete temporary working files after processing.\n"
        "Disable to re-run segmentation without repeating preprocessing."
    ),
    "minimise_output_size_mode": (
        "Delete non-essential output files to save storage space.\n"
        "Keeps only tree data CSVs and the report."
    ),

    # Model parameters (from other_parameters.py)
    "model_filename": (
        "Filename of the trained model weights (.pth file).\n"
        "Must be located in the model/ directory."
    ),
    "box_dimensions": (
        "Dimensions of the sliding box for semantic segmentation (metres).\n"
        "Default: [6, 6, 6]. Larger boxes provide more context but\n"
        "require more GPU memory."
    ),
    "box_overlap": (
        "Overlap fraction of the sliding box in each dimension.\n"
        "Default: [0.5, 0.5, 0.5]. Higher overlap improves accuracy\n"
        "but increases processing time."
    ),
    "min_points_per_box": (
        "Minimum number of points per box for model input.\n"
        "Boxes with fewer points are skipped."
    ),
    "max_points_per_box": (
        "Maximum number of points per box for model input.\n"
        "Boxes with more points are randomly subsampled.\n"
        "Higher values may require reducing batch_size."
    ),
    "grid_resolution": "Resolution of the Digital Terrain Model (DTM) grid in metres.",
    "min_cluster_size": (
        "Minimum cluster size for HDBSCAN clustering of stem points.\n"
        "Recommend leaving at default (30) for general use."
    ),
    "minimum_CCI": (
        "Minimum Circumferential Completeness Index for cylinder fitting.\n"
        "Measurements with CCI below this threshold are deleted.\n"
        "Range: 0.0 to 1.0. Default: 0.3."
    ),
    "min_tree_cyls": (
        "Minimum number of cylinders required per tree.\n"
        "Trees with fewer cylinders are deleted."
    ),
    "subsample": (
        "Enable point cloud subsampling before processing.\n"
        "Reduces point count to improve processing speed."
    ),
    "subsampling_min_spacing": (
        "Minimum distance between points after subsampling (metres).\n"
        "Default: 0.01 m (1 cm)."
    ),

    # Training parameters
    "epochs": (
        "Number of complete passes over the training dataset.\n"
        "Suggested: 500-2000 for fine-tuning, 2000-5000 for training from scratch.\n"
        "More epochs = longer training but potentially better accuracy."
    ),
    "learning_rate": (
        "Controls how much the model adjusts per training step.\n"
        "Suggested: 0.00001-0.0001. Lower values are more stable but slower.\n"
        "Default 0.000025 is a good starting point for fine-tuning."
    ),
    "train_batch_size": (
        "Number of point cloud boxes per training step.\n"
        "Higher values speed up training but use more GPU memory.\n"
        "Suggested: 2-8 depending on GPU VRAM. Reduce if you get OOM errors."
    ),
    "validation_batch_size": (
        "Number of point cloud boxes per validation step.\n"
        "Can be higher than training batch size since no gradients are stored.\n"
        "Suggested: 2-8."
    ),
    "class_weights": (
        "How to weight each class in the loss function during training.\n"
        "Auto: Compute inverse-frequency weights from training data.\n"
        "This gives underrepresented classes (e.g. CWD) higher weight,\n"
        "improving their accuracy significantly.\n"
        "None: Equal weight for all classes (original FSCT behaviour)."
    ),

    # Taper measurements
    "taper_measurement_height_min": "Lowest height for taper diameter measurements (metres above DTM).",
    "taper_measurement_height_max": "Highest height for taper diameter measurements (metres above DTM).",
    "taper_measurement_height_increment": "Height increment between taper measurements (metres).",
    "taper_slice_thickness": (
        "Thickness of slices used for taper measurements (metres).\n"
        "Cylinders within +/- half this thickness are considered.\n"
        "The largest diameter at each height is used."
    ),
}


def get_tooltip(param_name: str) -> str:
    """Get tooltip text for a parameter name."""
    return TOOLTIPS.get(param_name, "")
