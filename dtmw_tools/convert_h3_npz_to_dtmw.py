"""Convert H3-indexed trip records to the DT-MW simulator input format."""

import argparse
import os
import pickle
from datetime import datetime, timezone

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz-path",
        type=str,
        required=True,
        help="Prepared trip archive with H3 cell indices",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Processed dataset to write",
    )
    parser.add_argument("--start-date", type=str, default="2015-01-22")
    parser.add_argument("--end-date", type=str, default="2015-01-24")
    parser.add_argument("--max-distance", type=int, default=8)
    parser.add_argument("--target-max-orders", type=int, default=100)
    parser.add_argument(
        "--vip-rate",
        type=float,
        default=0.0,
        help="Fraction of trips receiving a fare multiplier",
    )
    parser.add_argument(
        "--vip-multiplier",
        type=float,
        default=10.0,
        help="Fare multiplier for selected trips",
    )
    parser.add_argument(
        "--vip-seed", type=int, default=42, help="Random seed for selecting VIP trips"
    )
    parser.add_argument(
        "--allow-pickled-npz",
        action="store_true",
        help="Allow legacy object arrays in a trusted NPZ",
    )
    args = parser.parse_args()
    if not 0 <= args.vip_rate <= 1 or args.vip_multiplier <= 0 or args.max_distance < 1:
        parser.error(
            "VIP rate must be in [0, 1]; VIP multiplier and max distance must be positive"
        )
    return args


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _choose_shape(n_cells: int) -> tuple[int, int]:
    for rows in range(int(np.sqrt(n_cells)), 1, -1):
        if n_cells % rows == 0:
            return rows, n_cells // rows
    raise ValueError("The simulator requires a rectangular grid with at least two rows and columns")


def build_neighbor(res8_h3: list[str], max_distance: int) -> np.ndarray:
    try:
        import h3
    except ImportError as exc:
        raise SystemExit(
            "Install requirements-optional.txt to use the H3 converter"
        ) from exc

    if not all(
        h3.is_valid_cell(cell) and h3.get_resolution(cell) == 8 for cell in res8_h3
    ):
        raise ValueError("res8_h3 must contain valid H3 resolution-8 cells")
    n_cells = len(res8_h3)
    neighbor = np.zeros((n_cells, n_cells), dtype=np.int32)
    for i, src in enumerate(res8_h3):
        for j in range(i + 1, n_cells):
            try:
                distance = min(h3.grid_distance(src, res8_h3[j]), max_distance)
            except (h3.H3FailedError, h3.H3GridNavigationError):
                distance = max_distance
            neighbor[i, j] = neighbor[j, i] = distance
    return neighbor


def main() -> None:
    args = parse_args()
    npz_path = os.path.abspath(args.npz_path)
    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with np.load(npz_path, allow_pickle=args.allow_pickled_npz) as data:
        origin_res8 = data["origin_res8"]
        dest_res8 = data["dest_res8"]
        fare_amount = data["fare_amount"]
        slot_offsets = data["slot_offsets"]
        slot_minutes = int(data["slot_minutes"])
        start_ts = int(data["start_ts"])
        res8_h3 = [str(x) for x in data["res8_h3"].tolist()]
        trip_time = data["trip_time"] if "trip_time" in data else None

    n_cells = len(res8_h3)
    if slot_minutes != 10:
        raise ValueError("The experiment simulator expects 10-minute slots")
    if not n_cells or not len(fare_amount) or len(set(res8_h3)) != n_cells:
        raise ValueError("Expected at least one fare and unique H3 cells")
    slots_per_day = 1440 // slot_minutes
    total_slots = len(slot_offsets) - 1

    start_date = _parse_date(args.start_date).date()
    end_date = _parse_date(args.end_date).date()
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    order_param_sum = np.zeros((n_cells, n_cells, slots_per_day), dtype=np.int64)
    day_count = 0

    real_orders_list = []

    start_day_idx = int(
        (start_date - datetime.fromtimestamp(start_ts, timezone.utc).date()).days
    )
    end_day_idx = int(
        (end_date - datetime.fromtimestamp(start_ts, timezone.utc).date()).days
    )

    shape = _choose_shape(n_cells)
    max_dist = min(args.max_distance, min(shape) - 1)
    neighbor = build_neighbor(res8_h3, max_dist)

    for day_idx in range(start_day_idx, end_day_idx + 1):
        day_slot_start = day_idx * slots_per_day
        day_slot_end = day_slot_start + slots_per_day
        if day_slot_start < 0 or day_slot_end > total_slots:
            continue
        for slot_in_day in range(slots_per_day):
            slot_idx = day_slot_start + slot_in_day
            start_idx = int(slot_offsets[slot_idx])
            end_idx = int(slot_offsets[slot_idx + 1])
            if end_idx <= start_idx:
                continue
            origin = origin_res8[start_idx:end_idx].astype(np.int64, copy=False)
            dest = dest_res8[start_idx:end_idx].astype(np.int64, copy=False)
            flat = origin * n_cells + dest
            counts = np.bincount(flat, minlength=n_cells * n_cells)
            order_param_sum[:, :, slot_in_day] += counts.reshape(n_cells, n_cells)

            slot_fare = fare_amount[start_idx:end_idx]
            for i in range(len(origin)):
                origin_grid = int(origin[i])
                dest_grid = int(dest[i])
                price = float(slot_fare[i])

                if trip_time is not None:
                    duration = max(1, int(trip_time[start_idx + i] // 600))
                else:
                    duration = max(1, int(neighbor[origin_grid, dest_grid]))

                real_orders_list.append(
                    [origin_grid, dest_grid, slot_in_day, duration, price]
                )
        day_count += 1

    if day_count == 0:
        raise RuntimeError("No slots found in the requested date range.")

    order_param = np.rint(order_param_sum / day_count).astype(np.int32)
    target_max = int(args.target_max_orders)
    if target_max > 0:
        order_num = order_param.sum(axis=1)
        current_max = int(order_num.max()) if order_num.size > 0 else 0
        if current_max > target_max:
            scale = target_max / float(current_max)
            order_param = np.floor(order_param.astype(np.float64) * scale).astype(
                np.int32
            )

    price_param = np.zeros((max_dist + 2, 2), dtype=np.float64)
    global_mean = float(np.mean(fare_amount))
    global_std = float(np.std(fare_amount)) if float(np.std(fare_amount)) > 0 else 1.0

    # Compute distance bins for all trips in range
    fare_list = []
    dist_list = []
    for day_idx in range(start_day_idx, end_day_idx + 1):
        day_slot_start = day_idx * slots_per_day
        day_slot_end = day_slot_start + slots_per_day
        if day_slot_start < 0 or day_slot_end > total_slots:
            continue
        start_idx = int(slot_offsets[day_slot_start])
        end_idx = int(slot_offsets[day_slot_end])
        if end_idx <= start_idx:
            continue
        origin = origin_res8[start_idx:end_idx].astype(np.int64, copy=False)
        dest = dest_res8[start_idx:end_idx].astype(np.int64, copy=False)
        dist = neighbor[origin, dest]
        dist = np.minimum(dist, max_dist)
        dist_list.append(dist)
        fare_list.append(fare_amount[start_idx:end_idx])

    if fare_list:
        dist_all = np.concatenate(dist_list)
        fare_all = np.concatenate(fare_list)
        for d in range(max_dist + 1):
            mask = dist_all == d
            if np.any(mask):
                mean = float(np.mean(fare_all[mask]))
                std = float(np.std(fare_all[mask]))
                price_param[d, 0] = mean
                price_param[d, 1] = std if std > 0 else global_std
            else:
                price_param[d, 0] = global_mean
                price_param[d, 1] = global_std
    else:
        price_param[:, 0] = global_mean
        price_param[:, 1] = global_std

    real_orders = np.asarray(real_orders_list, dtype=np.float64).reshape(-1, 5)

    if args.vip_rate > 0 and len(real_orders) > 0:
        vip_rng = np.random.default_rng(args.vip_seed)
        vip_mask = vip_rng.random(len(real_orders)) < args.vip_rate
        real_orders[vip_mask, 4] *= args.vip_multiplier
        vip_count = int(np.sum(vip_mask))
        print(
            f"VIP injection: {vip_count}/{len(real_orders)} orders ({100 * vip_count / len(real_orders):.1f}%) x{args.vip_multiplier} price"
        )

    data_param = {
        "neighbor": neighbor,
        "price": price_param,
        "order": order_param,
        "shape": shape,
        "real_orders": real_orders,
    }

    with open(out_path, "wb") as f:
        pickle.dump(data_param, f, protocol=4)

    print(f"Saved: {out_path}")
    print(f"Date range: {args.start_date} to {args.end_date} (days={day_count})")
    print(f"Grid cells: {n_cells}, shape: {shape}, slots/day: {slots_per_day}")
    print(f"Real orders: {len(real_orders)}")
    if len(real_orders) > 0:
        prices = real_orders[:, 4]
        print(f"  Price mean: {np.mean(prices):.2f}, std: {np.std(prices):.2f}")


if __name__ == "__main__":
    main()
