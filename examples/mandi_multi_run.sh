#!/bin/bash
# Production script for multi-run MANDI crystallography data reduction.
# Usage: ./mandi_multi_run.sh <directory_with_nexus_files> [output_dir]
set -e

if [[ -z "$1" ]]; then
    echo "Usage: $0 <directory_or_file> [output_dir]"
    exit 1
fi

if [[ -d "$1" ]]; then 
    DIR="$1"
    FILES=$DIR/*.nxs.h5
else 
    DIR="$(dirname "$1")"
    FILES=$@
fi

OUT_DIR=${2:-"reduction_output"}
mkdir -p "$OUT_DIR"

INSTRUMENT=MANDI
# Mesolite lattice parameters
INDEXER_PARAMS="18.39 56.55 6.54 90 90 90 Fdd2"

WAVEL_MIN=2.0
WAVEL_MAX=4.5
D_MIN=1.1

# 1. Reduce neutron counts of all runs (in parallel)
echo "--- Reducing runs ---"
FILE_LIST=()
for Z in $FILES; do
    BN=$(basename "$Z")
    python -m subhkl.io.parser reduce "$Z" "$OUT_DIR/$BN.reduce.h5" $INSTRUMENT &
    FILE_LIST+=("$OUT_DIR/$BN.reduce.h5")
done
wait

INTEGRATOR_THRESHOLDS="--region-growth-minimum-sigma 0.5 --region-growth-maximum-pixel-radius 12 --peak-center-box-size 3 --peak-smoothing-window-size 5 --peak-minimum-pixels 40 --peak-minimum-signal-to-noise 0.0 --peak-pixel-outlier-threshold 4"
FINDER_THRESHOLDS="--thresholding-noise-cutoff-quantile 0.99 --region-growth-minimum-intensity 3.0 --region-growth-maximum-pixel-radius 12.0 --peak-center-box-size 3 --peak-smoothing-window-size 5 --peak-minimum-pixels 40 --peak-minimum-signal-to-noise 0.0 --peak-pixel-outlier-threshold 4.0"

# 2. Merge images for peak finding
echo "--- Merging images ---"
python -m subhkl.io.parser merge-images "${FILE_LIST[*]}" "$OUT_DIR/scan_master.h5"

# 3. Find peaks
echo "--- Finding peaks ---"
XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false MIOPEN_DISABLE_CACHE=1 
python -m subhkl.io.parser finder "$OUT_DIR/scan_master.h5" $INSTRUMENT 
    --output-filename "$OUT_DIR/finder.h5" 
    --finder-algorithm thresholding $FINDER_THRESHOLDS 
    --create-visualizations --show-progress

# 4. Indexing (Two stages)
echo "--- Indexing Stage 1 (Coarse) ---"
python -m subhkl.io.parser indexer "$OUT_DIR/finder.h5" "$OUT_DIR/stage1.h5" $INDEXER_PARAMS 
    --wavelength-min $WAVEL_MIN --wavelength-max $WAVEL_MAX 
    --n-runs=10 --popsize=100 --gens=200 --strategy=de 
    --tolerance-deg=0.5 --loss-method gaussian 
    --hkl-search-range 35 --batch-size=1

echo "--- Indexing Stage 2 (Fine) ---"
python -m subhkl.io.parser indexer --bootstrap "$OUT_DIR/stage1.h5" "$OUT_DIR/finder.h5" "$OUT_DIR/indexer.h5" $INDEXER_PARAMS 
    --wavelength-min $WAVEL_MIN --wavelength-max $WAVEL_MAX 
    --n-runs=10 --popsize=100 --gens=250 --strategy=de 
    --tolerance-deg=0.1 --refine-lattice --lattice-bound-frac 0.05 
    --loss-method gaussian --hkl-search-range 35 --batch-size=1

python -m subhkl.io.parser metrics "$OUT_DIR/indexer.h5"

# 5. Predict peaks
echo "--- Predicting peaks ---"
python -m subhkl.io.parser peak-predictor "$OUT_DIR/scan_master.h5" $INSTRUMENT "$OUT_DIR/indexer.h5" "$OUT_DIR/peak_predictor.h5" 
    --wavel-min $WAVEL_MIN --wavel-max $WAVEL_MAX --d-min ${D_MIN}

# 6. Integrate peaks
echo "--- Integrating peaks ---"
python -m subhkl.io.parser integrator "$OUT_DIR/scan_master.h5" $INSTRUMENT "$OUT_DIR/peak_predictor.h5" "$OUT_DIR/integrator.h5" 
    --integration-method gaussian_fit --found-peaks-file "$OUT_DIR/finder.h5" 
    $INTEGRATOR_THRESHOLDS --create-visualizations 
    --peak-minimum-signal-to-noise 1.0 --show-progress

echo "--- Final Metrics ---"
python -m subhkl.io.parser metrics "$OUT_DIR/integrator.h5"
