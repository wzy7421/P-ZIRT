# P-ZIRT

P-ZIRT is a research prototype for provenance-aware zero-inflated roadside V2X proxy monitoring. It is intended for sparse lane-level or road-lane-level monitoring targets where most observations are zero and where decoded-message quality varies by source.

The current public code focuses on a synthetic demo and user-supplied CSV workflow. It does not include raw V2X files, manuscript drafts, generated review packages, or restricted derived tables.

## Main Idea

P-ZIRT separates a sparse monitoring target into two linked tasks:

1. nonzero proxy-event probability;
2. positive proxy magnitude when the event is nonzero.

The prototype also supports provenance/reliability weighting and road-lane group embeddings so that partially decoded or lower-quality samples do not have to be treated as equally reliable.

## Install

```bash
pip install -r requirements.txt
```

The main script requires Python 3.10+ and PyTorch.

## Run Synthetic Demo

```bash
python pzirt_model.py
```

The demo generates a synthetic roadside V2X-style table with sparse nonzero proxy values, provenance variables, and road-lane groups, then reports baseline and P-ZIRT metrics.

For a fast CPU smoke run:

```bash
python pzirt_model.py --epochs 2 --patience 1 --batch-size 512 --cpu
```

## Run With Your Own CSV

```bash
python pzirt_model.py --csv data.csv --target proxy \
  --group-col road_lane \
  --provenance-cols decoding_rate packet_quality \
  --split group
```

Expected columns:

- target column scaled to `[0, 1]`, for example `proxy`;
- group column, for example `road_lane`;
- numeric feature columns;
- optional provenance columns such as decoding completeness, packet quality, message coverage, or source availability.

Use `--feature-cols` to pass an explicit feature list. If omitted, the script uses numeric columns excluding target, group, and provenance columns.

## Reported Metrics

The script reports:

- RMSE and MAE for proxy magnitude;
- nonzero RMSE and MAE;
- PR-AUC and PR lift for rare-event ranking;
- Brier score, Brier skill, and ECE for probability calibration.

## Tests

```bash
pytest -q
```

The test suite runs a lightweight synthetic smoke test and checks that the core data preparation, baseline metrics, and short CPU training path execute successfully.

## Manuscript Alignment

The manuscript version associated with this project treats P-ZIRT as a reliability-oriented proxy-monitoring framework, not as validated physical queue-length estimation. Claims should remain limited to the data actually validated in a given study.

For formal publication, users should add independent validation such as video, detector, field logs, or manually audited queue-presence labels before claiming operational queue estimation.

## Data Policy

Raw roadside V2X records and manuscript review packages are intentionally not included. See [data/README.md](data/README.md) for the recommended data layout.

## License

No license has been assigned yet. Contact the repository owner before reusing this code beyond review or collaboration.
