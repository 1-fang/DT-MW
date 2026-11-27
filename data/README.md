# Data and experiment simulator

The CPU quick start generates its orders at runtime. This repository provides loaders and format converters for a researcher-supplied experiment input. It contains no trip-level records or trained checkpoints.

## Local experiment package

If you already have the original DT-MW experiment archive, unpack it outside this repository and set `DTMW_SIMULATOR_ROOT` to the unpacked `DTMW/` directory. The original simulator and loader are used through this directory. The corresponding CoopRide repository does not declare a redistribution license. [CoopRide source](https://github.com/tsinghua-fib-lab/CoopRide/tree/794186ad2f980a165f4397926eed9eacc2959eed)

DT-MW reads the following paths for experiment runs:

- `DTMW_SIMULATOR_ROOT` provides `dtmw_simulator/envs.py`, `dtmw_tools/load_data.py`, and `dtmw_tools/create_envs.py` along with their local dependencies.
- `DTMW_DATA_DIR` locates processed files referenced by the experiment launchers and weather configuration. It defaults to the experiment package's `data/` directory.
- `DTMW_DATA_PATH` and `DTMW_EVAL_DATA_PATH` select train and evaluation datasets.
- `DTMW_FEATURE_DIR` overrides the directory containing `feature_<grid_num>.pkl`. The experiment package's `assets/` is the default. If a feature-range file is unavailable, the worker falls back to runtime max normalization.

The checkpoint, feature-range, and processed data readers use Python pickle. Supply files from a source you trust.

| Experiment | Expected processed filename |
| --- | --- |
| NYC normal training | `NYC2015Jan05_h3_289_real_orders.pkl` |
| NYC normal evaluation | `NYC2015Jan06_h3_289_real_orders.pkl` |
| NYC VIP | `NYC2015Jan06_h3_289_vip.pkl` |
| NYC surge | `NYC2015Jan22-24_h3_289_real_orders.pkl` |
| NYC blizzard | `NYC2015Jan26_h3_289_real_orders.pkl` |
| DiDi Chengdu | `DiDi_chengdu_20161010_20161011_grid289_real_orders.pkl` |
| DiDi Xi'an | `DiDi_xian_day1_grid121_real_orders.pkl` |

The archive also contains legacy aggregate-order files named `NYU_grid143.pkl` and `DiDi_day1_grid121.pkl`. Use `DTMW_USE_REAL_ORDERS=0` with aggregate-only data.

Obtain NYC and DiDi records under the respective providers' data access terms. For the NYC weather schedule, use a consistent H3 cell-to-index mapping across cohorts. [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) · [DiDi GAIA](https://gaia.didichuxing.com/en)

## Processed format

Each input is a dictionary with a `neighbor` matrix of grid distances, a `price` array of fare means and standard deviations, an `order` tensor of shape `[origin, destination, 144]`, and optional `shape` and `duration` metadata. Real-order playback uses the `real_orders` array. Its first five columns are:

```text
origin_grid, destination_grid, start_slot, duration_slots, fare
```

A day has 144 ten-minute slots. The origin and destination indices refer to the topology for that run.

## Conversion tools

The H3 converter takes a prepared NPZ with `origin_res8`, `dest_res8`, `fare_amount`, `slot_offsets`, `slot_minutes`, `start_ts`, and `res8_h3`. The first two are integer indices into `res8_h3`. Optional `trip_time` gives durations in seconds. Dates are inclusive and computed from the UTC start timestamp. The command assumes the raw data have already been filtered and mapped to this input schema.

```bash
python -m pip install -r requirements-optional.txt
python -m dtmw_tools.convert_h3_npz_to_dtmw \
  --npz-path local-data/nyc_january_h3.npz \
  --out-path local-data/NYC2015Jan05_h3_289_real_orders.pkl \
  --start-date 2015-01-05 --end-date 2015-01-05
```

The DiDi converter takes a prepared NPZ with `origin_grid`, `dest_grid`, and `slot_offsets`. Optional arrays include `trip_time` (seconds) and `fare_amount`. For a non-11×11 grid, provide `grid_rows` and `grid_cols`. It folds complete days into a 144-slot profile. The base pickle supplies the neighborhood, duration, and pricing parameters.

```bash
python -m dtmw_tools.convert_didi_npz_to_real_orders_pkl \
  --npz local-data/didi_prepared.npz \
  --base-pkl local-data/didi_base.pkl \
  --output local-data/didi_real_orders.pkl
```
