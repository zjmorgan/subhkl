INTEGRATOR_PARAMS="--region-growth-minimum-intensity 25 --region-growth-maximum-pixel-radius 6 --peak-center-box-size 3 --peak-smoothing-window-size 5 --peak-minimum-pixels 40 --peak-minimum-signal-to-noise 4.0"
python -m subhkl.io.parser finder $1 MANDI --output-filename $1.finder.h5 --finder-algorithm thresholding --thresholding-noise-cutoff-quantile 0.99 $INTEGRATOR_PARAMS
python -m subhkl.io.parser indexer de 4 10 0.9 $1.finder.h5 $1.indexer.h5 18.39 56.55 6.54 90 90 90 2 4.5 F --seed 12345
python -m subhkl.io.parser peak-predictor $1 MANDI $1.indexer.h5 $1.peak_predictor.h5 --d-min 1.35
python -m subhkl.io.parser integrator $1 MANDI $1.peak_predictor.h5 $1.integrator.h5 $INTEGRATOR_PARAMS
python -m subhkl.io.parser mtz-exporter $1.integrator.h5 $1.mtz "F d d 2"
python -c "import reciprocalspaceship as rs; ds = rs.read_mtz('$1.mtz'); print(ds.head()); print(f'{len(ds)} reflections')"
