# DT-MW

**Dual-Timescale LLM-Guided Manager-Worker Reinforcement Learning for Ride-Hailing Dispatching and Repositioning**

Camera-ready code for **Findings of WSDM 2027**. A low-frequency manager interprets scenario context and proposes a natural-language instruction and a semantic prior. A dual controller corrects the prior with observed order response and driver-income fairness. A high-frequency MAPPO worker uses the corrected reward and the instruction to dispatch and reposition vehicles. [Conference track](https://www.wsdm-conference.org/2027/cffp.html) · [Method overview](docs/METHOD_OVERVIEW.md)

## Quick start

Use Python 3.10 or 3.11. The smoke test runs on CPU using generated trips and the bundled instruction embeddings.

```bash
git clone https://github.com/1-fang/DT-MW.git
cd DT-MW
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/run_train_smoke.sh
```

The script trains for two 24-step episodes, loads the resulting checkpoint, and writes summary and per-step evaluation CSVs to a new `logs/smoke.*` directory. The metrics describe the generated example.

## Run the ride-hailing experiments

These commands require a local copy of the original DT-MW experiment simulator and the processed datasets. Set the simulator root to the directory holding `dtmw_simulator/` and the legacy `dtmw_tools/`. The [data and simulator guide](data/README.md) lists the expected files, conversion commands, and data sources.

```bash
export DTMW_SIMULATOR_ROOT="$PWD/../dtmw-experiment/DTMW"
export DTMW_DATA_DIR="$DTMW_SIMULATOR_ROOT/data"
export DTMW_DATA_PATH="$DTMW_DATA_DIR/NYC2015Jan05_h3_289_real_orders.pkl"
export DTMW_GRID_NUM=289 DTMW_DRIVER_NUM=2000
export DTMW_USE_REAL_ORDERS=1 DTMW_REAL_ORDER_SAMPLE_RATE=0.35
export DTMW_MAX_ITER=400 DTMW_SAVE_INTERVAL=20
export DTMW_LOG_DIR="$PWD/logs/nyc"
python -m run.train
```

Evaluation defaults to the January 6 dataset in the same data directory. Point `DTMW_RESUME_PATH` to a checkpoint file or a training output directory.

```bash
DTMW_RESUME_PATH="$DTMW_LOG_DIR" \
DTMW_EVAL_DATA_PATH="$DTMW_DATA_DIR/NYC2015Jan06_h3_289_real_orders.pkl" \
bash scripts/run_eval_example.sh
```

The evaluation writes `result/eval_outputs/<tag>_<timestamp>.csv` and its companion `_detail.csv`. The summary includes gross merchandise value (GMV), order response rate (ORR), and driver-income Gini. Set `DTMW_EVAL_REPEATS` for additional evaluation seeds and `DTMW_EVAL_GREEDY=1` for greedy dispatch. Use `DTMW_EVAL_OUTPUT` to set the summary path.

| Setting | Use |
| --- | --- |
| `DTMW_SKIP_WEATHER=1` | Restrict training to the dataset in `DTMW_DATA_PATH`. Use this for DiDi. |
| `DTMW_WEATHER_CONFIG` | Select a custom multi-day schedule. The default NYC schedule is [weather_days.json](config/weather_days.json). |
| `DTMW_TRAIN_WEATHER_TEXT`, `DTMW_TRAIN_EVENT_TEXT`, `DTMW_TRAIN_TRAFFIC_TEXT` | Supply single-day manager context. |
| `DTMW_EVAL_WEATHER_TEXT` | Override the evaluation scenario description inferred from the dataset filename. |
| `DTMW_SEED` and `DTMW_RUN_TAG` | Seed a run and label its output directory. |
| `DTMW_CPU=1` | Run without CUDA. The default device is the first visible CUDA GPU when available. |
| `DTMW_LR` | Set the actor and critic learning rates. The launcher defaults to `1e-3`. |

### Ablations

Run `bash scripts/run_ablation_example.sh` with the same simulator and dataset setup. Set a distinct `DTMW_RUN_TAG` when launching variants individually. The batch script creates a directory per variant under `DTMW_LOG_DIR` when set.

| Ablation | Setting |
| --- | --- |
| Dual feedback | `DTMW_NO_DUAL=1` |
| Semantic prior | `DTMW_USE_LLM_PRIOR=0` |
| Semantic state augmentation | `DTMW_NO_SSA=1` |
| Dynamic reward shaping | `DTMW_DRS_WEIGHT=0` |
| Intrinsic semantic alignment | `DTMW_INTRINSIC_WEIGHT=0 DTMW_CONTRASTIVE_WEIGHT=0` |

### Online manager and custom embeddings

The default manager is a deterministic scenario lookup over the bundled configurations. To generate priors with an online model, install `requirements-optional.txt`, set `DTMW_USE_LLM=1`, and provide `OPENAI_API_KEY` through your environment. `DTMW_MODEL` selects the model and `OPENAI_BASE_URL` can point to an OpenAI-compatible service. For new instruction text, the semantic encoder loads the configured SBERT model or a local `DTMW_SBERT_SNAPSHOT`. The bundled MiniLM cache covers all nine texts in the stock scenario and weather files. [Embedding model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Citation and license

Please cite *DT-MW: Dual-Timescale LLM-Guided Manager-Worker Reinforcement Learning for Ride-Hailing Dispatching and Repositioning*, Findings of WSDM 2027, when using this code.

The DT-MW code in this repository is available under the [MIT License](LICENSE). The release includes the DT-MW manager, worker, training and evaluation code, an independent synthetic environment, scenario configuration, and conversion tools. The data guide describes the external simulator, datasets, and feature ranges used for the paper experiments. [CoopRide upstream](https://github.com/tsinghua-fib-lab/CoopRide/tree/794186ad2f980a165f4397926eed9eacc2959eed)
