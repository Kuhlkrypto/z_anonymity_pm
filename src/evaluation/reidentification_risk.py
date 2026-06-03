from pm4py.objects.log.obj import EventLog
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import random
import numpy as np
from typing import List, Any, Tuple
from math import ceil
from datetime import datetime
from dateutil import parser  # ensure python-dateutil is available


def _is_subset(small: Counter, big: Counter) -> bool:
    for k, v in small.items():
        if big.get(k, 0) < v:
            return False
    return True


def _sample_indices(trace_length: int, number_points: float, rng: random.Random) -> List[int]:
    if trace_length == 0:
        return []
    if number_points > 1:
        k = int(number_points)
    else:
        k = max(1, int(trace_length * number_points))
    if trace_length <= k:
        return list(range(trace_length))
    return rng.sample(range(trace_length), k)


def _extract_ngrams(sequence: List[Any], n: int) -> List[Tuple[Any, ...]]:
    if len(sequence) < n:
        return [tuple(sequence)]
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def _sample_ngrams(ngrams: List[Tuple[Any, ...]], number_points: float, rng: random.Random) -> List[Tuple[Any, ...]]:
    total = len(ngrams)
    if total == 0:
        return []
    if total == 1:
        return ngrams[:]
    if number_points > 1:
        k = int(number_points)
        if total <= k:
            return ngrams[:]
        return rng.sample(ngrams, k)
    else:
        k = max(1, int(ceil(total * number_points)))
        if total <= k:
            return ngrams[:]
        return rng.sample(ngrams, k)


def _list_is_subset(small_list, big_list):
    # frequency-aware subset check like unicity_activites.check_subset
    if not all(elem in big_list for elem in small_list):
        return False
    big_vals, big_counts = np.unique(big_list, return_counts=True)
    small_vals, small_counts = np.unique(small_list, return_counts=True)
    for val, cnt in zip(small_vals, small_counts):
        matches = np.where(big_vals == val)[0]
        if len(matches) == 0:
            return False
        if big_counts[matches[0]] < cnt:
            return False
    return True


# ---------------------------------------------------------------------------
# Module-level state shared across Pool workers (set once via initializer)
# ---------------------------------------------------------------------------
_reid_full_counter_act = None
_reid_full_counter_ts  = None


def _reid_worker_init(full_counter_act, full_counter_ts):
    """Pool initializer: copy precomputed Counters into each worker process."""
    global _reid_full_counter_act, _reid_full_counter_ts
    _reid_full_counter_act = full_counter_act
    _reid_full_counter_ts  = full_counter_ts


def _reid_check_trace(args):
    """
    Check a single sampled pattern against its pre-filtered candidate traces.
    Top-level picklable function required by multiprocessing.Pool.
    """
    i, pat_act, pat_ts, candidates = args
    pat_act_counter = Counter(pat_act)
    pat_ts_counter  = Counter(pat_ts)
    match_count = 0
    matched_indices = []
    for j in sorted(candidates):
        if not _is_subset(pat_act_counter, _reid_full_counter_act[j]):
            continue
        if not _is_subset(pat_ts_counter, _reid_full_counter_ts[j]):
            continue
        match_count += 1
        matched_indices.append(j)
    return i, match_count, (match_count == 1), matched_indices


def detailed_counts_projection_A(traces_act, traces_time, number_points, seed=None, n_jobs=1):
    """
    List-based projection A (activities + timestamps) in the style of unicity_activites.py.
    Returns per-trace match counts, uniqueness flags, and overall unicity fraction.
    number_points: absolute integer count of points to sample per trace.

    Optimizations applied:
    - Inverted index reduces the candidate set for each pattern from O(n_traces) to
      O(|traces containing all sampled activities|), dramatically cutting comparisons.
    - n_jobs > 1 parallelises the per-trace matching loop across a multiprocessing.Pool;
      Counter state is shared via the Pool initializer to avoid per-task pickling.
    """
    if seed is not None:
        random.seed(seed)

    n_traces = len(traces_act)
    sampled_patterns = []

    # Sample indices per trace using absolute number_points
    for i in range(n_traces):
        acts = traces_act[i]
        ts = traces_time[i]
        trace_len = len(acts)
        if trace_len == 0:
            idxs = []
        else:
            k = int(number_points)
            if trace_len <= k:
                idxs = list(range(trace_len))
            else:
                idxs = random.sample(list(range(trace_len)), k)

        sampled_act = [acts[j] for j in idxs if j < len(acts) and acts[j] is not None]
        sampled_ts  = [ts[j]  for j in idxs if j < len(ts)  and ts[j]  is not None]
        sampled_patterns.append({'activity': sampled_act, 'timestamp': sampled_ts})

    # Precompute Counter representations once to avoid O(n^2) re-building
    full_counter_act = [Counter(a for a in acts  if a is not None) for acts  in traces_act]
    full_counter_ts  = [Counter(t for t in times if t is not None) for times in traces_time]

    # Build inverted index: activity -> set of trace indices that contain it
    act_to_traces: defaultdict = defaultdict(set)
    for j, acts in enumerate(traces_act):
        for a in set(a for a in acts if a is not None):
            act_to_traces[a].add(j)

    # Precompute candidate sets per pattern via index intersection
    task_args = []
    for i, pat in enumerate(sampled_patterns):
        unique_acts = set(pat['activity'])
        if unique_acts:
            candidates = set.intersection(*(act_to_traces[a] for a in unique_acts))
        else:
            candidates = set(range(n_traces))
        task_args.append((i, pat['activity'], pat['timestamp'], candidates))

    per_trace_results = [None] * n_traces
    uniques = 0

    if n_jobs > 1:
        # Parallel path: distribute per-trace checks across Pool workers.
        # Counters are shared via initializer (one copy per worker, not per task).
        with Pool(
            processes=n_jobs,
            initializer=_reid_worker_init,
            initargs=(full_counter_act, full_counter_ts),
        ) as pool:
            for i, match_count, is_unique, matched_indices in pool.imap_unordered(
                _reid_check_trace, task_args
            ):
                if is_unique:
                    uniques += 1
                per_trace_results[i] = {
                    "sampled_activity": sampled_patterns[i]['activity'],
                    "sampled_timestamp": sampled_patterns[i]['timestamp'],
                    "match_count": match_count,
                    "is_unique": is_unique,
                    "matched_trace_indices": matched_indices,
                }
    else:
        # Sequential path: inline check with index-filtered candidates
        for i, pat_act, pat_ts, candidates in task_args:
            pat_act_counter = Counter(pat_act)
            pat_ts_counter  = Counter(pat_ts)
            match_count = 0
            matched_indices = []
            for j in sorted(candidates):
                if not _is_subset(pat_act_counter, full_counter_act[j]):
                    continue
                if not _is_subset(pat_ts_counter, full_counter_ts[j]):
                    continue
                match_count += 1
                matched_indices.append(j)
            is_unique = (match_count == 1)
            if is_unique:
                uniques += 1
            per_trace_results[i] = {
                "sampled_activity": pat_act,
                "sampled_timestamp": pat_ts,
                "match_count": match_count,
                "is_unique": is_unique,
                "matched_trace_indices": matched_indices,
            }

    unicity_fraction = uniques / n_traces if n_traces > 0 else 0.0
    return {
        "per_trace": per_trace_results,
        "unicity_fraction": unicity_fraction,
    }


def calculate_reidentification_risk(
    log: EventLog,
    projection: str = 'A_list*',  # 'N' = n-gram, 'E' = activities only, 'A' = activities + timestamps, 'A_list' = list-based A, 'A*'/'A_list*' = day-granular variants
    number_points: float = 5,  # >=1 absolute, <1 fraction (also governs number of n-grams in N)
    repetitions: int = 10,
    seed: int | None = None,
    n_jobs: int = 1,
    ngram_size: int = 7  # used only when projection == 'N'
) -> dict:
    """
    Compute unicity (re-identification risk) under projections:
      - 'E': activities only,
      - 'A': activities + aligned timestamps (Counter-based),
      - 'A_list': activities + timestamps using list-based subset logic (unicity_activites style),
      - 'A*' / 'A_list*': same as 'A' / 'A_list' but with timestamps reduced to day granularity,
      - 'N': n-gram uniqueness (sampled n-grams per trace).
    """
    # Normalize projection variants
    use_list_based = False
    if projection.endswith('*'):
        use_day_granular = True
        base_proj = projection[:-1]
    else:
        use_day_granular = False
        base_proj = projection

    if base_proj == 'A_list':
        use_list_based = True
        norm_projection = 'A_list'
    elif base_proj == 'A':
        norm_projection = 'A'
    elif base_proj == 'E':
        norm_projection = 'E'
    elif base_proj == 'N':
        norm_projection = 'N'
    else:
        raise ValueError(f"Unknown projection: {projection}")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    base_rng = random.Random(seed if seed is not None else random.randrange(2**30))

    # Extract per-trace sequences
    traces_act: List[List[Any]] = []
    traces_time: List[List[Any]] = []
    for trace in log:
        activities = [event.get('concept:name') for event in trace]
        timestamps = [event.get('time:timestamp') for event in trace]
        traces_act.append(activities)
        traces_time.append(timestamps)

    # Handle day-granular reduction if requested
    if use_day_granular:
        def reduce_to_day(ts):
            if ts is None:
                return None
            if isinstance(ts, datetime):
                dt = ts
            else:
                try:
                    dt = datetime.fromisoformat(ts)
                except Exception:
                    dt = parser.parse(ts)
            return datetime(dt.year, dt.month, dt.day)

        traces_time = [
            [reduce_to_day(ts) for ts in trace_times]
            for trace_times in traces_time
        ]
        # keep norm_projection as 'A' or 'A_list' depending on original

    n_traces = len(traces_act)
    if n_traces == 0:
        key = f'{norm_projection}_{ngram_size}_{number_points}_points' if norm_projection == 'N' else f'{norm_projection}_{number_points}_points'
        return {
            'parameters': {
                'projection': projection,
                'number_points': number_points if norm_projection in ('A', 'E', 'A_list') else None,
                'repetitions': repetitions,
                'seed': seed,
                'n_jobs': n_jobs,
                'ngram_size': ngram_size if norm_projection == 'N' else None
            },
            'risk_metrics': {
                key: {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'individual_runs': [0.0] * (repetitions if norm_projection != 'N' else 1)
                }
            }
        }

    # Precompute full counters for activity and timestamp (for Counter-based A/E)
    full_counter_activity = [Counter([a for a in acts if a is not None]) for acts in traces_act]
    full_counter_timestamp = None
    if norm_projection == 'A':
        full_counter_timestamp = [Counter([t for t in times if t is not None]) for times in traces_time]

    def _single_rep_AE(rep_idx: int) -> float:
        rng = random.Random(base_rng.randint(0, 2**30 - 1) + rep_idx)

        sampled_patterns = []
        for i in range(n_traces):
            acts = traces_act[i]
            ts = traces_time[i]
            trace_len = len(acts)

            idxs = _sample_indices(trace_len, number_points, rng)

            sampled_act = Counter([acts[j] for j in idxs if j < len(acts) and acts[j] is not None])
            pattern = {'activity': sampled_act}

            if norm_projection in ('A',):
                sampled_ts = Counter([ts[j] for j in idxs if j < len(ts) and ts[j] is not None])
                pattern['timestamp'] = sampled_ts

            sampled_patterns.append(pattern)

        uniques = 0
        memo = {}
        for i, pat in enumerate(sampled_patterns):
            key = (
                tuple(sorted(pat['activity'].items())),
                tuple(sorted(pat['timestamp'].items())) if norm_projection == 'A' else None
            )
            if key in memo:
                uniques += memo[key]
                continue

            match_count = 0
            for j in range(n_traces):
                if not _is_subset(pat['activity'], full_counter_activity[j]):
                    continue
                if norm_projection == 'A':
                    if not _is_subset(pat['timestamp'], full_counter_timestamp[j]):
                        continue
                match_count += 1
                if match_count > 1:
                    break
            is_unique = 1 if match_count == 1 else 0
            memo[key] = is_unique
            uniques += is_unique

        return uniques / n_traces

    def _single_rep_N(rep_idx: int) -> float:
        rng = random.Random(base_rng.randint(0, 2**30 - 1) + rep_idx)

        # Build full n-grams per trace (filtered of None)
        trace_ngrams = []
        for acts in traces_act:
            cleaned = [a for a in acts if a is not None]
            ngrams = _extract_ngrams(cleaned, ngram_size)
            trace_ngrams.append(ngrams)

        # Build inverted index of all n-grams to traces containing them
        gram_to_traces = defaultdict(set)
        for idx, grams in enumerate(trace_ngrams):
            for g in grams:
                gram_to_traces[g].add(idx)

        # Sample per-trace subset of its n-grams according to number_points
        sampled_trace_ngrams = []
        for i, grams in enumerate(trace_ngrams):
            sampled = _sample_ngrams(grams, number_points, rng)
            sampled_trace_ngrams.append(sampled)

        # Determine uniqueness: any sampled gram exclusive to trace
        uniques = 0
        for i, sampled_grams in enumerate(sampled_trace_ngrams):
            found_unique = False
            for g in sampled_grams:
                owners = gram_to_traces.get(g, set())
                if len(owners) == 1 and i in owners:
                    found_unique = True
                    break
            if found_unique:
                uniques += 1
        return uniques / n_traces

    # Dispatch logic
    if norm_projection == 'N':
        if n_jobs == 1:
            results = [_single_rep_N(i) for i in range(repetitions)]
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                results = list(ex.map(_single_rep_N, range(repetitions)))
        metric_key = f'N_{ngram_size}_{number_points}_points'
        arr = np.array(results)
        return {
            'parameters': {
                'projection': projection,
                'number_points': None,
                'repetitions': repetitions,
                'seed': seed,
                'n_jobs': n_jobs,
                'ngram_size': ngram_size
            },
            'risk_metrics': {
                'mean': float(arr.mean()),
                'std': float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
                'min': float(arr.min()),
                'max': float(arr.max()),
                'individual_runs': [float(x) for x in results],
            }
        }

    if norm_projection == 'A_list':
        # Compute absolute number of points analogous to unicity_activites logic
        max_len = max(len(t) for t in traces_act) if traces_act else 0
        if number_points <= 1:
            abs_points = ceil(number_points * max_len)
        else:
            abs_points = int(number_points)

        # Single draw to mirror original behavior
        detailed = detailed_counts_projection_A(traces_act, traces_time, abs_points, seed=seed, n_jobs=n_jobs)
        unicity_fraction = detailed["unicity_fraction"]
        return {
            'parameters': {
                'projection': projection,
                'number_points': number_points,
                'repetitions': 1,
                'seed': seed,
                'n_jobs': 1,
                'ngram_size': None
            },
            'risk_metrics': {
                'mean': unicity_fraction,
                'std': 0.0,
                'min': unicity_fraction,
                'max': unicity_fraction,
                'individual_runs': [unicity_fraction],
                'per_trace': detailed['per_trace'],
            }
        }

    # Standard 'A' or 'E' (Counter-based)
    if n_jobs == 1:
        results = [_single_rep_AE(i) for i in range(repetitions)]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_single_rep_AE, range(repetitions)))
    arr = np.array(results)
    metric_key = f'{norm_projection}_{number_points}_points'
    return {
        'parameters': {
            'projection': projection,
            'number_points': number_points if norm_projection in ('A', 'E', 'A_list') else None,
            'repetitions': repetitions,
            'seed': seed,
            'n_jobs': n_jobs,
            'ngram_size': ngram_size if norm_projection == 'N' else None
        },
        'risk_metrics': {
            'mean': float(arr.mean()),
            'std': float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
            'min': float(arr.min()),
            'max': float(arr.max()),
            'individual_runs': [float(x) for x in results],
        }
    }
