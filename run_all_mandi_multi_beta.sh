WAVEL_MIN=2
D_MIN=1.9
WAVEL_MAX=4
INTEGRATOR_PARAMS="--region-growth-minimum-intensity 35 --region-growth-maximum-pixel-radius 12 --peak-center-box-size 3 --peak-smoothing-window-size 5 --peak-minimum-pixels 40 --peak-minimum-signal-to-noise 1.0 --peak-pixel-outlier-threshold 2"
FINDER_PARAMS="104.339 104.339 98.975 90 90 120 ${WAVEL_MIN} ${WAVEL_MAX} P"

# per h5
FIND="python -m subhkl.io.parser finder \$FN MANDI --output-filename \$FN.finder.h5 --finder-algorithm thresholding --thresholding-noise-cutoff-quantile 0.98 $INTEGRATOR_PARAMS --create-visualizations --thresholding-mask-file mandi_mask.png --thresholding-mask-rel-erosion-radius 0.1"

# single call
INDEX=$(cat <<EOF
python -m subhkl.io.parser finder-merger $1/finder_files.txt $1/finder_merged.h5 $FINDER_PARAMS;
python -m subhkl.io.parser indexer $1/finder_merged.h5 $1/stage1.h5 $FINDER_PARAMS --n-runs=5 --popsize=1000 --gens=100 --strategy=de --softness=0.1 --refine-goniometer --goniometer-bound-deg=15 --refine-lattice;
python -m subhkl.io.parser indexer --bootstrap $1/stage1.h5 $1/finder_merged.h5 $1/indexer.h5 $FINDER_PARAMS --n-runs=5 --popsize=1000 --gens=50 --strategy=de --softness=1e-3 --refine-goniometer --goniometer-bound-deg=2;
EOF
)
#python -m subhkl.io.parser indexer --bootstrap $1/stage2.h5 $1/finder_merged.h5 $1/indexer.h5 $FINDER_PARAMS --n-runs=5 --popsize=1000 --gens=100 --strategy=de --softness=5e-3;

# per h5
INTEGRATE=$(cat <<EOF
python -m subhkl.io.parser peak-predictor \$FN MANDI $1/indexer.h5 \$FN.peak_predictor_multi.h5 --d-min ${D_MIN};
python -m subhkl.io.parser integrator \$FN MANDI \$FN.peak_predictor_multi.h5 \$FN.integrator_multi.h5 $INTEGRATOR_PARAMS --create-visualizations --show-progress;
python -m subhkl.io.parser mtz-exporter \$FN.integrator_multi.h5 \$FN.mtz "P 64 2 2";
python -c "import reciprocalspaceship as rs; ds = rs.read_mtz('\$FN.mtz'); print(ds.head()); print(f'{len(ds)} reflections')";
EOF
)

#for Z in $1/*nxs.h5; do
###    FN=$Z srun -n 1 -N1 -c 1 --gpus-per-task=1 bash -c "$FIND" &
    #FN=$Z bash -c "$FIND" &
#done
wait

#ls $1/*.finder.h5 > $1/finder_files.txt
#eval $INDEX

for Z in $1/*nxs.h5; do
#    FN=$Z srun -n 1 -N1 -c 1 --gpus-per-task=1 bash -c "$INTEGRATE" &
    FN=$Z bash -c "$INTEGRATE" &
done
wait
