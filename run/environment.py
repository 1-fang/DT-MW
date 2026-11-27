"""Select DT-MW's bundled synthetic example or a local experiment simulator."""

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def configure_environment(args, default_grid):
    args.synthetic = os.getenv("DTMW_SYNTHETIC") == "1"
    default_grid = 9 if args.synthetic else default_grid
    args.grid_num = int(
        os.getenv("DTMW_GRID_NUM", os.getenv("COOPRIDE_GRID_NUM", default_grid))
    )
    driver_defaults = {9: 36, 100: 1000, 121: 1500, 143: 2000, 289: 2000}
    default_drivers = driver_defaults.get(args.grid_num, 2000)
    args.driver_num = int(
        os.getenv("DTMW_DRIVER_NUM", os.getenv("COOPRIDE_DRIVER_NUM", default_drivers))
    )
    args.env_seed = {100: 16, 121: 6, 143: 326}.get(args.grid_num, 0)
    if args.driver_num < 1:
        raise ValueError("DTMW_DRIVER_NUM must be positive")
    if args.synthetic:
        args.TIME_LEN = int(os.getenv("DTMW_EPISODE_STEPS", args.TIME_LEN))
        if args.TIME_LEN < 1:
            raise ValueError("DTMW_EPISODE_STEPS must be positive")


def legacy_root():
    value = os.getenv("DTMW_SIMULATOR_ROOT")
    return Path(value).expanduser().resolve() if value else None


def create_environment(args):
    if args.synthetic:
        if args.custom_data_path or args.use_real_orders:
            raise ValueError(
                "Synthetic runs require DTMW_USE_REAL_ORDERS=0 and no dataset path"
            )
        from run.synthetic import SyntheticDispatchEnv

        return SyntheticDispatchEnv(args.grid_num, args.driver_num)

    root = legacy_root()
    if root is None:
        raise ValueError(
            "Set DTMW_SIMULATOR_ROOT to a local DT-MW experiment simulator, "
            "or run scripts/run_train_smoke.sh for the synthetic example."
        )
    required = (
        "dtmw_simulator/envs.py",
        "dtmw_tools/load_data.py",
        "dtmw_tools/create_envs.py",
    )
    for relative_path in required:
        if not (root / relative_path).is_file():
            raise FileNotFoundError(
                f"Experiment simulator file is missing: {root / relative_path}"
            )

    sys.path.append(str(root))
    from dtmw_tools.create_envs import create_OD, create_OD_36
    from dtmw_tools.load_data import (
        load_envs_custom,
        load_envs_DiDi121,
        load_envs_NYU143,
    )

    if args.custom_data_path:
        dataset = Path(args.custom_data_path).expanduser().resolve(strict=True)
        args.custom_data_path = str(dataset)
        env, _, _, _, args.grid_num = load_envs_custom(
            dataset,
            driver_num=args.driver_num,
            use_real_orders=args.use_real_orders,
            real_order_sample_rate=args.real_order_sample_rate,
        )
    elif args.grid_num == 100:
        env, _, _, _, args.grid_num = create_OD()
    elif args.grid_num == 36:
        env, _, _, _, args.grid_num = create_OD_36()
    elif args.grid_num == 121:
        env, _, _, _, args.grid_num = load_envs_DiDi121(driver_num=args.driver_num)
    elif args.grid_num == 143:
        env, _, _, _, args.grid_num = load_envs_NYU143(driver_num=args.driver_num)
    else:
        raise ValueError(
            f"Set a processed dataset path for the {args.grid_num}-grid environment"
        )

    env.fleet_help = True
    return env
