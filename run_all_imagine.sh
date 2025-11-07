python -m subhkl.io.parser finder $1 IMAGINE --output-filename $1.finder.h5 --peak-local-max-min-relative-intensity 0.05 --peak-local-max-min-pixel-distance 15 --peak-local-max-normalization
python -m subhkl.io.parser indexer de 4 $1.finder.h5 $1.indexer.h5 18.39 56.55 6.54 90 90 90 2 4.5 F
python -m subhkl.io.parser peak-predictor $1 IMAGINE $1.indexer.h5 $1.peak_predictor.h5 --d-min 1.35
python -m subhkl.io.parser integrator $1 IMAGINE $1.peak_predictor.h5 $1.integrator.h5
python -m subhkl.io.parser mtz-exporter $1.integrator.h5 $1.mtz "F d d 2"