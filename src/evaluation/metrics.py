import email.utils
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional, Any

import pm4py
from pm4py import generalization_tbr, Marking, PetriNet
from pm4py.objects.log.obj import EventLog
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

from typing import Optional
from pm4py.objects.log.obj import EventLog
from pm4py.discovery import discover_petri_net_inductive
from pm4py.conformance import fitness_alignments, fitness_token_based_replay # Fitness assessment
from pm4py.conformance import precision_alignments, precision_token_based_replay # Precision assessment
from pm4py import simplicity_petri_net


def get_alphabet(log: EventLog) -> Set[str]:
    alphabet = set()
    for trace in log:
        for event in trace:
            if hasattr(event, "get"):
                activity = event.get("concept:name")
            elif isinstance(event, dict):
                activity = event.get("concept:name")
            else:
                activity = event
            if activity:
                alphabet.add(activity)
    return alphabet


def compute_eventual_follows_relations(
    log: EventLog, alphabet: Set[str]
) -> Dict[str, Dict[str, int]]:
    trace_counts = {
        a: {b: {"a_occurs": 0, "b_after": 0} for b in alphabet if b != a}
        for a in alphabet
    }

    for trace in log:
        trace_activities = [
            evt.get("concept:name") if hasattr(evt, "get") else evt for evt in trace
        ]
        for a in alphabet:
            if a not in trace_activities:
                continue
            last_a_idx = len(trace_activities) - 1 - trace_activities[::-1].index(a)
            for b in alphabet:
                if a == b:
                    continue
                trace_counts[a][b]["a_occurs"] += 1
                if any(
                    trace_activities[i] == b
                    for i in range(last_a_idx + 1, len(trace_activities))
                ):
                    trace_counts[a][b]["b_after"] += 1

    result = {a: {} for a in alphabet}
    for a in alphabet:
        for b in alphabet:
            if a == b:
                continue
            a_occurs = trace_counts[a][b]["a_occurs"]
            b_after = trace_counts[a][b]["b_after"]
            if a_occurs == 0:
                relation = 0
            elif b_after == 0:
                relation = 0
            elif b_after == a_occurs:
                relation = 2
            else:
                relation = 1
            result[a][b] = relation
    return result


def calculate_f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def compare_eventual_follows_relations(
    original_log: EventLog, generated_log: EventLog
) -> Dict[str, Any]:
    original_alphabet = get_alphabet(original_log)
    original_rel = compute_eventual_follows_relations(original_log, original_alphabet)

    support = {"never": 0, "sometimes": 0, "always": 0}
    for a in original_alphabet:
        for b in original_alphabet:
            if a == b:
                continue
            rel = original_rel[a][b]
            if rel == 0:
                support["never"] += 1
            elif rel == 1:
                support["sometimes"] += 1
            elif rel == 2:
                support["always"] += 1

    if len(generated_log) == 0:
        return {
            "support": support,
            "per_class": {
                "never": {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None},
                "sometimes": {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None},
                "always": {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None},
            },
            "micro": {"precision": None, "recall": None, "f1": None},
            "macro": {"precision": None, "recall": None, "f1": None},
        }

    generated_alphabet = get_alphabet(generated_log)
    generated_rel = compute_eventual_follows_relations(generated_log, generated_alphabet)

    class_map = {0: "never", 1: "sometimes", 2: "always"}
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for a in original_alphabet:
        for b in original_alphabet:
            if a == b:
                continue
            orig = original_rel[a][b]
            gen = (
                generated_rel[a][b]
                if (a in generated_alphabet and b in generated_alphabet)
                else 0
            )
            if orig == gen:
                tp[class_map[orig]] += 1
            else:
                fn[class_map[orig]] += 1
                fp[class_map[gen]] += 1

    per_class_metrics = {}
    for cls in ["never", "sometimes", "always"]:
        cls_tp = tp[cls]
        cls_fp = fp[cls]
        cls_fn = fn[cls]
        precision = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0.0
        recall = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0.0
        f1 = calculate_f1(cls_tp, cls_fp, cls_fn)
        per_class_metrics[cls] = {
            "tp": cls_tp,
            "fp": cls_fp,
            "fn": cls_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = calculate_f1(total_tp, total_fp, total_fn)

    precisions = [per_class_metrics[c]["precision"] for c in ["never", "sometimes", "always"]]
    recalls = [per_class_metrics[c]["recall"] for c in ["never", "sometimes", "always"]]
    f1s = [per_class_metrics[c]["f1"] for c in ["never", "sometimes", "always"]]
    macro_precision = sum(precisions) / 3
    macro_recall = sum(recalls) / 3
    macro_f1 = sum(f1s) / 3

    return {
        "support": support,
        "per_class": per_class_metrics,
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
    }


# ---------- New Metrics with updated naming ----------

def get_directly_follows_relations(log: EventLog) -> Set[Tuple[str, str]]:
    """
    Extract set of directly-follows (a, b) pairs from the log (immediate successors).
    """
    df = set()
    for trace in log:
        activities = [
            evt.get("concept:name") if hasattr(evt, "get") else evt for evt in trace
        ]
        for i in range(len(activities) - 1):
            a, b = activities[i], activities[i + 1]
            if a is not None and b is not None:
                df.add((a, b))
    return df


def get_ratio_of_remaining_directly_follows(
    reference_log: EventLog, anonymized_log: EventLog
) -> Optional[float]:
    """
    Ratio of directly-follows relations from reference_log that remain in anonymized_log.
    Returns None if reference_log has no directly-follows pairs.
    """
    df_ref = get_directly_follows_relations(reference_log)
    if not df_ref:
        return None
    df_anonymized = get_directly_follows_relations(anonymized_log)
    preserved = df_ref.intersection(df_anonymized)
    ratio = len(preserved) / len(df_ref)
    return ratio


def compute_fitness(
    anonymized_log: EventLog, net, im, fm, alignment_based:bool, multi_processing: bool
) -> Optional[float]:
    """
    Mines a Petri net from reference_log using the inductive miner (with optional noise threshold)
    and computes the token-based replay fitness of anonymized_log against that model.

    Returns the average trace fitness (float in [0,1]) or None if something fails.
    """
    if len(anonymized_log) == 0:
        return None
    try:

        if alignment_based:
            fitness_dict = fitness_alignments(anonymized_log, net, im, fm, multi_processing=multi_processing)
        else:
            # Get the log-level token-based replay fitness
            fitness_dict = fitness_token_based_replay(
                anonymized_log, net, im, fm
            )
        # fitness_token_based_replay and fitness_alignments returns keys like 'average_trace_fitness'
        return fitness_dict.get("average_trace_fitness", None)

    except Exception:
        return None



def compute_precision(
    anonymized_log: EventLog, net, im, fm, alignment_based: bool, muli_processing: bool
) -> Optional[float]:
    """
    Computes the precision of the model w.r.t. the anonymized log.


    Returns precision of the model w.r.t. the anonymized log or None if something fails.

    """

    try:
        if alignment_based:
            return precision_alignments(anonymized_log, net, im, fm, multi_processing=muli_processing)

        return precision_token_based_replay(anonymized_log, net, im, fm)

    except Exception:
        return None

def discover_model(log: EventLog, muli_processing=False) -> Optional[tuple[PetriNet, Marking, Marking]]:
    try:
        return discover_petri_net_inductive(log, multi_processing=muli_processing)
    except Exception as e:
        print(e)
        return None

def compute_generality(reference_log, net, im, fm) -> Optional[float]:
    try:
        return generalization_tbr(reference_log, net, im, fm)
    except Exception:
        return None

def compute_simplicity(net, im, fm) -> Optional[float]:
    try:
        return simplicity_petri_net(net, im, fm)
    except Exception:
        return None

def compute_quality_metrics(
    reference_log: EventLog, anonymized_log: EventLog, alignment_based: bool = False, multi_processing: bool = False
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:

    if len(anonymized_log) == 0:
        print("no anonymized log")
        print(anonymized_log)
        return None, None, None, None

    try:
        print("[DEBUG]: Started computing quality metrics")
        # Discover model using anonymized log
        model = discover_model(anonymized_log, multi_processing)
        print("[DEBUG]: Finished computing model")

        if model is None:
            return None, None, None, None
        net, im, fm = model

        # compute fitness of the model against anonymized log
        # fitness = compute_fitness(reference_log, net, im, fm, alignment_based, multi_processing)
        print("[DEBUG]: Finished computing fitness")

        # compute precision of the model against anonymized log
        precision = compute_precision(reference_log, net, im, fm, alignment_based, multi_processing)
        print("[DEBUG]: Finished computing precision")
        # model = discover_model(anonymized_log, multi_processing)
        # if model is None:
        #     return fitness, precision, None, None

        # net, im, fm = model

        # compute generalization
        # generalization = compute_generality(reference_log, net, im, fm)
        print("[DEBUG]: Finished computing generalization")

        # compute simplicity of 'anonymized model'
        # simplicity = compute_simplicity(net, im, fm)
        print("[DEBUG]: Finished computing Simplicity")



        # return fitness, precision, generalization, simplicity
        return precision

    except Exception:
        return None, None, None, None

