import os
import json
import random
from typing import Literal
from pathlib import Path
import multiprocessing
from multiprocessing import Pool

import numpy as np
from pm4py.algo.evaluation.replay_fitness.variants import alignment_based

from evaluation.metrics import compute_quality_metrics
from src.utils.log_utils import load_event_log
from src.definitions import get_output_path
from src.anonymization.z_anonymity import apply_z_anonymity
from src.anonymization.ngram_z_anonymity import apply_ngram_z_anonymity
from src.anonymization.baseline import apply_baseline
from src.evaluation.metrics import get_ratio_of_remaining_directly_follows, compute_fitness
from src.evaluation.event_log_stats import get_event_log_stats
from src.evaluation.reidentification_risk import calculate_reidentification_risk

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVENT_LOG_PATH = Path(os.path.join(SCRIPT_DIR, '../data/input/')).resolve()

# Globals for worker caching
_worker_original_log = None
_worker_log_path = None
_worker_source_attribute = None


def anonymize_log(original_log, z, time_window, mode='single', ngram_size=1, explicit=False, source_attribute=None):
    if mode == 'single':
        return apply_z_anonymity(original_log, z, time_window, explicit, source_attribute)
    else:  # mode == 'ngram'
        return apply_ngram_z_anonymity(original_log, z, time_window, ngram_size, explicit, source_attribute)


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def _worker_init(log_path: str, source_attribute: str):
    """Initializer for Pool workers: load the log once per worker."""
    global _worker_original_log, _worker_log_path, _worker_source_attribute
    if _worker_log_path != log_path:
        _worker_original_log = load_event_log(log_path)
        _worker_log_path = log_path
        _worker_source_attribute = source_attribute


def _run_single_task(args):
    """
    Task executed in worker (Pool mode). Expects a dict with keys:
      z, time_window, mode, ngram_size, explicit, log_name, algorithm, seed
    Uses cached original log loaded in initializer.
    """
    z = args['z']
    time_window = args['time_window']
    mode = args['mode']
    ngram_size = args['ngram_size']
    explicit = args['explicit']
    log_name = args['log_name']
    algorithm = args['algorithm']
    seed = args.get('seed', None)

    # alignment based fitness and precision computation
    alignment_based = args.get('alignment_based', False)

    # In Pool mode multiprocessing is always False (outer parallelism used instead)
    use_multiprocessing = False
    global _worker_original_log, _worker_source_attribute
    original_log = _worker_original_log
    source_attribute = _worker_source_attribute

    return _execute_task(
        original_log, source_attribute,
        z, time_window, mode, ngram_size, explicit,
        log_name, algorithm, seed, alignment_based, use_multiprocessing
    )


def _execute_task(
    original_log, source_attribute,
    z, time_window, mode, ngram_size, explicit,
    log_name, algorithm, seed, alignment_based, use_multiprocessing
):
    """
    Core computation shared between Pool-worker and sequential execution paths.
    """
    # Apply anonymization or baseline
    if algorithm == 'baseline':
        anonymized_log, number_of_edited_traces = apply_baseline(
            original_log, z, time_window if time_window is not None else 0, mode, ngram_size, explicit
        )
    else:
        if mode == 'single':
            anonymized_log, number_of_edited_traces = apply_z_anonymity(
                original_log, z, time_window if time_window is not None else 0, explicit, source_attribute
            )
        else:  # 'ngram'
            anonymized_log, number_of_edited_traces = apply_ngram_z_anonymity(
                original_log, z, time_window if time_window is not None else 0, ngram_size, explicit, source_attribute
            )

    anonymized_log_stats = get_event_log_stats(original_log, anonymized_log, number_of_edited_traces)
    print("[DEBUG]: Finished computing anonymized log stats")
    ratio_of_remaining_directly_follows = get_ratio_of_remaining_directly_follows(original_log, anonymized_log)
    print("[DEBUG]: Finished computing ratio of remaining directly follows")
    # compute the quality metrics – multiprocessing flag controls internal pm4py parallelism
    fitness, precision, generalization, simplicity = compute_quality_metrics(
        original_log, anonymized_log, alignment_based, use_multiprocessing
    )
    print("[DEBUG]: Finished computing quality metrics")
    if len(anonymized_log) == 0:
        reidentification_protection = None
        print("[DEBUG]: RISK is None")
    else:
        # Determine base projection
        projection_base = 'A_list*'

        # When running sequentially (use_multiprocessing=True), pm4py is done here
        # and all cores are free → use them for the risk calculation.
        # In Pool mode the outer pool owns the cores, so keep n_jobs=1 to avoid
        # nested process explosion.
        n_jobs_risk = multiprocessing.cpu_count() if use_multiprocessing else 1
        reid_risk = calculate_reidentification_risk(
            anonymized_log, projection=projection_base, ngram_size=ngram_size,
            seed=seed, n_jobs=n_jobs_risk
        )
        print("[DEBUG]: Finished computing RISK")
        reidentification_protection = 1 - reid_risk['risk_metrics']['mean']

    result = {
        'parameters': {
            'anom_alg': algorithm,
            'z': z,
            'time_window': time_window,
            'mode': mode,
            'ngram_size': ngram_size if mode == 'ngram' else None,
            'explicit': explicit
        },
        'anonymized_log_stats': anonymized_log_stats,
        'ratio_of_remaining_directly_follows': ratio_of_remaining_directly_follows,
        'fitness': fitness,
        'precision': precision,
        'generalization': generalization,
        'simplicity': simplicity,
        'reidentification_protection': reidentification_protection
    }
    return result


def _flush_results(results: list, output_path: str) -> None:
    """Serialize and write current results list to disk."""
    print("[DEBUG]: About to flush results.")
    serializable_results = convert_to_serializable(results)
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def test_different_z_values_with_pool(
    log_path: str,
    time_windows: list,
    z_values: list,
    mode: Literal['single', 'ngram'] = 'single',
    ngram_size: int = 3,
    explicit: bool = False,
    source_attribute: str = None,
    log_name: str = None,
    cores_to_use: int = 4,
    seed: int | None = None,
    alignment_based: bool = False,
    multi_processing: bool = False,
):
    mode_suffix = f"_{mode}"
    if mode == 'ngram':
        mode_suffix += f"_ngram{ngram_size}"
    if explicit:
        mode_suffix += "_explicit"
    output_filename = f'results{mode_suffix}.json'
    output_path = get_output_path(output_filename, log_name)

    # Prepare task list
    tasks = []
    for z in z_values:
        for window in time_windows:
            tasks.append({
                'log_path': log_path,
                'z': z,
                'time_window': window,
                'mode': mode,
                'ngram_size': ngram_size,
                'explicit': explicit,
                'source_attribute': source_attribute,
                'log_name': log_name,
                'algorithm': 'z-anonymity',
                'seed': seed,
                'alignment_based': alignment_based,
            })
    for z in z_values:  # baseline tasks
        tasks.append({
            'log_path': log_path,
            'z': z,
            'time_window': None,
            'mode': mode,
            'ngram_size': ngram_size,
            'explicit': explicit,
            'source_attribute': source_attribute,
            'log_name': log_name,
            'algorithm': 'baseline',
            'seed': seed,
            'alignment_based': alignment_based,
        })

    results = []

    if multi_processing:
        # -------------------------------------------------------
        # Sequential outer loop so pm4py's internal parallelism
        # (alignment computation, process discovery) can utilise
        # all available cores without contention.
        # -------------------------------------------------------
        print(
            f"[multi_processing=True] Running {len(tasks)} tasks sequentially; "
            "pm4py internal parallelism enabled."
        )
        original_log = load_event_log(log_path)
        for i, task in enumerate(tasks, 1):
            print(
                f"  [{i}/{len(tasks)}] alg={task['algorithm']} "
                f"z={task['z']} window={task['time_window']}"
            )
            result = _execute_task(
                original_log, source_attribute,
                task['z'], task['time_window'], task['mode'],
                task['ngram_size'], task['explicit'],
                task['log_name'], task['algorithm'],
                task['seed'], task['alignment_based'],
                use_multiprocessing=True,          # enable pm4py-internal MP
            )
            results.append(result)
            # Write results after every completed task
            _flush_results(results, output_path)
            print(f"     -> result written to {output_path}")
    else:
        # -------------------------------------------------------
        # Parallel outer loop: spawn one process per task,
        # pm4py internal parallelism disabled inside workers.
        # -------------------------------------------------------
        total_cores = multiprocessing.cpu_count()
        cores_to_use = max(1, int(total_cores * 0.8))
        print(
            f"[multi_processing=False] Running {len(tasks)} tasks in Pool "
            f"with {cores_to_use} workers."
        )
        with Pool(
            processes=cores_to_use,
            initializer=_worker_init,
            initargs=(log_path, source_attribute)
        ) as pool:
            for result in pool.imap_unordered(_run_single_task, tasks):
                results.append(result)
                # Write results after every completed task
                _flush_results(results, output_path)

    return results


def main():
    raw_logs = [f for f in os.listdir(EVENT_LOG_PATH) if f.endswith(".xes.gz")]
    for raw_log in raw_logs:
        log_name = os.path.splitext(os.path.splitext(raw_log)[0])[0]
        if log_name == 'Sepsis' or log_name.__contains__("Hospital"):
            source_attribute = 'org:group'
        else:
            source_attribute = 'org:resource'
        full_path = os.path.join(EVENT_LOG_PATH, raw_log)

        h = 3600 # 1 hour in seconds
        TIME_WINDOWS = [4*h, 24*h, 48*h, 72*h]  # 72h in seconds
        Z_VALUES = list(range(1, 31))
        for mode in ['ngram']:  # you can add 'single' if desired
            for ngram_size in [1,2,3]:
                for explicit in [False, True]:
                    print(
                        f"\nTesting z-anonymity for {log_name} | "
                        f"mode={mode} | ngram_size={ngram_size} | explicit={explicit}"
                    )
                    test_different_z_values_with_pool(
                        full_path,
                        TIME_WINDOWS,
                        Z_VALUES,
                        mode=mode,
                        ngram_size=ngram_size,
                        explicit=explicit,
                        source_attribute=source_attribute,
                        log_name=log_name,
                        cores_to_use=4,  # adjust for your machine
                        seed=42,
                        alignment_based=True,
                        multi_processing=True,
                    )


if __name__ == '__main__':
    main()
