import copy
from collections import defaultdict
from typing import List, Set, Tuple

from pm4py.objects.log.obj import EventLog


def extract_ngrams_from_trace(trace_activity_list, n):
    """
    Return list of tuples: (ngram_string, start_pos, end_pos_inclusive)
    """
    if len(trace_activity_list) < n:
        return []
    ngrams = []
    for i in range(len(trace_activity_list) - n + 1):
        ngram = ">".join(trace_activity_list[i : i + n])
        # positions [i, i+n-1]
        ngrams.append((ngram, i, i + n - 1))
    return ngrams


def apply_baseline(
    original_log: EventLog,
    z: int,
    window,  # ignored: the baseline uses Δt = ∞ by design
    mode: str,
    ngram_size: int,
    explicit: bool,
) -> Tuple[EventLog, int]:
    """
    Centralized baseline anonymization.

    Corresponds to running the same operators on the collapsed (single-stream)
    log with Δt = ∞, i.e. without any source partition and without a time
    window. Matches of the chosen behavior B are processed in completion-time
    order across the whole log; ties at the exact same completion time are
    handled in arrival order, matching the streaming convention of the
    per-source filters (and of the original z-anonymity algorithm of
    Jha et al., 2020).

    mode: "single"  -> B^a (single activity occurrences).
          "ngram"   -> B^w (consecutive n-grams over the trace).

    explicit=False (zanon with Δt = ∞):
        A match (c, τ, E) is released iff, after adding c to the set of cases
        that have shown behavior B by completion time τ, the total number of
        distinct contributing cases is ≥ z. The first z-1 distinct cases to
        exhibit B are therefore suppressed; everything from the z-th case
        onward is released.

    explicit=True  (ezanon with Δt = ∞):
        Once at least z distinct cases have ever exhibited behavior B, every
        match of B is released, including those from the first z-1 cases that
        completed before the threshold was reached.
    """
    if mode not in {"single", "ngram"}:
        raise ValueError(f"Unknown mode '{mode}'; expected 'single' or 'ngram'")
    if mode == "ngram" and ngram_size <= 0:
        raise ValueError("ngram_size must be >=1 for ngram mode")

    anonymized_log = copy.deepcopy(original_log)
    edited_traces: Set[int] = set()

    # Collect, per behavior key, the list of matches.
    # A match record is (completion_timestamp, case_id, positions) where
    # `positions` are positions in the case's own trace.
    behavior_matches: defaultdict[str, List[Tuple]] = defaultdict(list)

    if mode == "single":
        for trace in original_log:
            case_id = trace.attributes["concept:name"]
            for pos, ev in enumerate(trace):
                act = ev["concept:name"]
                ts = ev["time:timestamp"]
                # singleton match: positions tuple has one element
                behavior_matches[act].append((ts, case_id, (pos,)))
    else:  # mode == "ngram"
        for trace in original_log:
            case_id = trace.attributes["concept:name"]
            activities = [ev["concept:name"] for ev in trace]
            timestamps = [ev["time:timestamp"] for ev in trace]
            if len(activities) < ngram_size:
                continue
            for i in range(0, len(activities) - ngram_size + 1):
                ngram = ">".join(activities[i : i + ngram_size])
                completion_ts = timestamps[i + ngram_size - 1]
                positions = tuple(range(i, i + ngram_size))
                behavior_matches[ngram].append((completion_ts, case_id, positions))

    # Per-behavior release computation.
    events_to_release: Set[Tuple[str, int]] = set()

    for matches in behavior_matches.values():
        # Sort by completion timestamp; Python's sort is stable, so ties at
        # the same timestamp are processed in their original (arrival) order.
        matches.sort(key=lambda m: m[0])

        # Quick rejection: behaviors that never reach z distinct cases in the
        # whole log cannot qualify under either operator.
        distinct_cases = {case_id for _, case_id, _ in matches}
        if len(distinct_cases) < z:
            continue

        if explicit:
            # ezanon, Δt = ∞: once the threshold is met, every match of this
            # behavior is released, in every case.
            for _, case_id, positions in matches:
                for pos in positions:
                    events_to_release.add((case_id, pos))
        else:
            # zanon, Δt = ∞: sweep matches in completion-time order, release
            # only those processed after the running set of distinct cases has
            # reached z.
            seen_cases: Set[str] = set()
            for _, case_id, positions in matches:
                seen_cases.add(case_id)
                if len(seen_cases) >= z:
                    for pos in positions:
                        events_to_release.add((case_id, pos))

    # Apply filtering: keep only events whose (case_id, pos) is in the
    # release set; drop now-empty traces.
    traces_to_delete: List[int] = []
    for trace_idx, trace in enumerate(anonymized_log):
        case_id = trace.attributes["concept:name"]
        kept = []
        for pos, ev in enumerate(trace):
            if (case_id, pos) in events_to_release:
                kept.append(ev)
            else:
                edited_traces.add(trace_idx)
        trace._list = kept
        if not kept:
            traces_to_delete.append(trace_idx)
    for ti in sorted(traces_to_delete, reverse=True):
        del anonymized_log._list[ti]

    return anonymized_log, len(edited_traces)
