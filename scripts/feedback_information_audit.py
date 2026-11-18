"""Audit saved feedback and rule choices; never run a solver, model, or final data.

This is a retrospective diagnostic, not a new discovery experiment. The grammar
was fixed by the earlier corpus protocol; per-state best-rule numbers use labels
in hindsight and must not be reported as a deployable policy's performance.
"""
import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from tsp_value.data import distances, length, validate_tour
from tsp_value.programs import canonical_program, compile_rule, INITIAL
from tsp_value.solver import select_edge

EPS = 1e-7  # percentage points; same measured-preference threshold as R1


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def restore_observables(row):
    """Reconstruct numeric observables only, without starting any search."""
    state = copy.deepcopy(row["state"])
    for key in ("coords", "tour", "best_tour", "penalties"):
        state[key] = np.asarray(state[key], dtype=float if key == "coords" else np.int64)
    state["distance"] = distances(state["coords"])
    return state


def classify(value):
    return "positive" if value > EPS else "negative" if value < -EPS else "tie"


def rank_correlation(x, y):
    """Spearman with average ranks; undefined if either variable is constant."""
    def ranks(values):
        a = np.asarray(values, dtype=float)
        order = np.argsort(a, kind="stable")
        r = np.empty(len(a), dtype=float)
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a[order[j]] == a[order[i]]:
                j += 1
            r[order[i:j]] = (i + j - 1) / 2
            i = j
        return r
    rx, ry = ranks(x), ranks(y)
    return None if np.ptp(rx) == 0 or np.ptp(ry) == 0 else float(np.corrcoef(rx, ry)[0, 1])


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.perf_counter(), time.process_time()
    inputs = set()

    def tracked(path):
        inputs.add(path)
        return read(path)

    bankpath = CODE / "results/core/prepare-v1/bank_observables.jsonl"
    inputs.add(bankpath)
    bank = [json.loads(line) for line in bankpath.read_text(encoding="utf-8").splitlines()]
    exact = {r["bank_id"]: r for r in
             (tracked(p) for p in sorted((CODE / "results/core/prepare-v1/exact_states").glob("*.json")))}
    assert len(bank) == len(exact) == 32
    cfg = tracked(CODE / "configs/core_abc_v1.json")
    grammar = tracked(CODE / "results/corpus-v1/grammar_candidates.json")
    tracked(CODE / "configs/core_corpus_v1.json")
    groups = {"old_grammar": {}, "r1_generated": {}}
    for program in grammar:
        p = canonical_program(program["source"])
        assert p["program_sha256"] == program["program_sha256"]
        groups["old_grammar"][p["program_sha256"]] = {**p, "family": program["family"]}
    for path in sorted((CODE / "results/core/discovery-v1/programs").glob("*.py")):
        inputs.add(path)
        p = canonical_program(path.read_text(encoding="utf-8"))
        assert p["program_sha256"] == path.stem
        groups["r1_generated"][path.stem] = {**p, "family": "r1_generated"}
    allprograms = {key: p for group in groups.values() for key, p in group.items()}
    programs = list(allprograms.values())
    rules = {p["program_sha256"]: compile_rule(p["expression"]) for p in programs}
    baseline = canonical_program(INITIAL)["program_sha256"]
    assert baseline in allprograms

    choices, values, state_rows, witnesses = {}, {}, [], []
    revalued_branch_routes = revalued_endpoint_routes = 0
    max_route_error = 0.0
    for row in bank:
        state = restore_observables(row)
        edges = list(map(tuple, row["edges"]))
        measured = np.asarray(exact[row["bank_id"]]["true_scores_pp"])
        assert edges == [tuple(r["edge"]) for r in exact[row["bank_id"]]["records"]]
        assert abs(measured[-1]) <= EPS
        for record, value in zip(exact[row["bank_id"]]["records"], measured):
            validate_tour(record["best_tour"], row["nodes"])
            tour_cost = length(np.asarray(record["best_tour"], dtype=np.int64), state["distance"])
            max_route_error = max(max_route_error, abs(tour_cost - record["best_length"]))
            expected = 100 * (exact[row["bank_id"]]["records"][-1]["best_length"] - tour_cost) / state["best_length"]
            assert abs(tour_cost - record["best_length"]) < 1e-10
            assert abs(expected - value) < 1e-10
            revalued_branch_routes += 1
        for p in programs:
            key = p["program_sha256"]
            edge = select_edge(copy.deepcopy(state), rules[key])
            choices[key, row["bank_id"]] = edge
            values[key, row["bank_id"]] = float(measured[edges.index(edge)])
        cap = cfg["delta_budget_by_nodes"][str(row["nodes"])]
        sr = dict(bank_id=row["bank_id"], instance_id=row["instance_id"], nodes=row["nodes"],
                  state_index=row["state_index"], work_used=state["delta_evaluations"],
                  inside_whole_program_cap=state["delta_evaluations"] < cap,
                  all_actions=len(edges), positive_actions=int(np.sum(measured > EPS)),
                  negative_actions=int(np.sum(measured < -EPS)), best_action_value_pp=float(measured.max()),
                  baseline_value_pp=values[baseline, row["bank_id"]])
        for groupname, group in groups.items():
            vv = [values[k, row["bank_id"]] for k in group]
            sr[groupname + "_positive_rules"] = sum(v > EPS for v in vv)
            sr[groupname + "_nonzero_rules"] = sum(abs(v) > EPS for v in vv)
            sr[groupname + "_best_reachable_pp"] = max(vv)
            sr[groupname + "_distinct_chosen_edges"] = len({choices[k, row["bank_id"]] for k in group})
        state_rows.append(sr)
        d = state["distance"][state["tour"], np.roll(state["tour"], -1)]
        edge_penalties = state["penalties"][state["tour"], np.roll(state["tour"], -1)]
        for idx in np.flatnonzero(measured > EPS):
            edge = edges[int(idx)]
            hits = {g: [k for k in group if choices[k, row["bank_id"]] == edge]
                    for g, group in groups.items()}
            # A label-informed, state-specific existence witness, NOT a newly
            # discovered general rule. Peaking at this observed distance tests
            # whether the current arithmetic interface can express this choice.
            target = float(state["distance"][edge] / d.mean())
            witness = canonical_program(f"-(d / mean_d - {target!r}) ** 2")
            witness_edge = select_edge(copy.deepcopy(state), compile_rule(witness["expression"]))
            witnesses.append(dict(bank_id=row["bank_id"], instance_id=row["instance_id"],
                nodes=row["nodes"], state_index=row["state_index"], edge=list(edge), value_pp=float(measured[idx]),
                distance=float(state["distance"][edge]),
                distance_over_mean=target,
                penalty=int(state["penalties"][edge]), reference_edge=row["reference_edge"],
                longer_edges_with_same_penalty=int(np.sum((edge_penalties == state["penalties"][edge]) &
                                                         (d > state["distance"][edge]))),
                old_grammar_rule_count=len(hits["old_grammar"]),
                r1_generated_rule_count=len(hits["r1_generated"]),
                example_grammar_expression=allprograms[hits["old_grammar"][0]]["expression"] if hits["old_grammar"] else None,
                hindsight_state_specific_expression=witness["expression"],
                hindsight_witness_selects_target=witness_edge == edge))

    # Replay *choices*, not solver trajectories, against every saved R1 decision.
    decision_count = 0
    for path in sorted((CODE / "results/core/discovery-v1/decisions").glob("*.json")):
        obj = tracked(path)
        assert obj["ok"] and path.stem in allprograms
        for r in obj["results"]:
            assert tuple(r["edge"]) == choices[path.stem, r["bank_id"]]
            decision_count += 1

    # Match existing whole-rule endpoints on the SAME eight original instances.
    evaluations = {}
    for path in sorted((CODE / "results/core/discovery-v1/evaluations").glob("discovery-gls-*.json")):
        obj = tracked(path)
        if obj["ok"]:
            evaluations[obj["program_sha256"]] = obj
    base_results = {(r["instance_id"], r["seed"]): r for r in evaluations[baseline]["results"]}
    instance_ids = sorted({s["instance_id"] for s in bank})
    original_coords = {r["instance_id"]: r["state"]["coords"] for r in bank}
    matched, program_rows = [], []
    for p in programs:
        key = p["program_sha256"]
        local = [values[key, s["bank_id"]] for s in bank]
        pr = dict(program_sha256=key, expression=p["expression"],
                  in_old_grammar=key in groups["old_grammar"], in_r1_generated=key in groups["r1_generated"],
                  mean_local_value_pp=float(np.mean(local)), positive_states=sum(v > EPS for v in local),
                  nonzero_states=sum(abs(v) > EPS for v in local),
                  distinct_action_vector_sha256=hashlib.sha256(json.dumps(
                      [choices[key, s["bank_id"]] for s in bank]).encode()).hexdigest(),
                  matched_endpoint_original_instances=0, mean_matched_full_rule_benefit_gap_pp=None,
                  mean_all_discovery_full_rule_benefit_gap_pp=None)
        if key in evaluations:
            result = {(r["instance_id"], r["seed"]): r for r in evaluations[key]["results"]}
            for iid in instance_ids:
                rr = [r for (rid, seed), r in result.items() if rid == iid]
                assert len(rr) == 2 and {r["seed"] for r in rr} == {0, 1}
                dm = distances(original_coords[iid])
                for r in rr:
                    validate_tour(r["best_tour"], r["nodes"])
                    tour_cost = length(np.asarray(r["best_tour"], dtype=np.int64), dm)
                    max_route_error = max(max_route_error, abs(tour_cost - r["original_length"]))
                    assert abs(tour_cost - r["original_length"]) < 1e-10
                    assert abs(100 * (tour_cost / r["reference_length"] - 1) - r["gap_percent"]) < 1e-10
                    assert r["delta_evaluations"] == cfg["delta_budget_by_nodes"][str(r["nodes"])]
                    revalued_endpoint_routes += 1
                states = [s for s in bank if s["instance_id"] == iid]
                localvals = [values[key, s["bank_id"]] for s in states]
                gain = float(np.mean([base_results[iid, r["seed"]]["gap_percent"] - r["gap_percent"] for r in rr]))
                matched.append(dict(program_sha256=key, instance_id=iid, nodes=rr[0]["nodes"],
                    local_states=len(states), mean_local_value_pp=float(np.mean(localvals)),
                    any_nonzero_local=any(abs(v) > EPS for v in localvals),
                    full_rule_benefit_gap_pp=gain,
                    endpoint_seeds=[0, 1], bank_trajectory_seed=17))
            mm = [r for r in matched if r["program_sha256"] == key]
            pr["matched_endpoint_original_instances"] = len(mm)
            pr["mean_matched_full_rule_benefit_gap_pp"] = float(np.mean([r["full_rule_benefit_gap_pp"] for r in mm]))
            pr["mean_all_discovery_full_rule_benefit_gap_pp"] = float(np.mean([
                base_results[k]["gap_percent"] - r["gap_percent"] for k, r in result.items()]))
        program_rows.append(pr)

    group_summary = {}
    for name, group in groups.items():
        vv = np.asarray([[values[k, s["bank_id"]] for s in bank] for k in group])
        group_summary[name] = dict(programs=len(group), program_state_decisions=int(vv.size),
            positive_decisions=int(np.sum(vv > EPS)), negative_decisions=int(np.sum(vv < -EPS)),
            nonzero_decisions=int(np.sum(np.abs(vv) > EPS)),
            positive_states_reachable=int(np.sum(np.any(vv > EPS, axis=0))),
            states_with_any_nonzero_reachable=int(np.sum(np.any(np.abs(vv) > EPS, axis=0))),
            per_state_best_rule_mean_pp=float(vv.max(axis=0).mean()),
            max_single_rule_mean_pp=float(vv.mean(axis=1).max()),
            distinct_action_vectors=len({r["distinct_action_vector_sha256"] for r in program_rows if r["program_sha256"] in group}))
    paired = [r for r in program_rows if r["matched_endpoint_original_instances"]]
    localx = [r["mean_local_value_pp"] for r in paired]
    fully = [r["mean_matched_full_rule_benefit_gap_pp"] for r in paired]
    summary = dict(scope="retrospective saved-data audit; no solver/model/LLM calls; no new data or final access",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        measured_value_epsilon_pp=EPS, bank_original_instances=len(instance_ids), bank_states=len(bank),
        all_edge_measured_positive_states=sum(s["positive_actions"] > 0 for s in state_rows),
        all_edge_measured_nonzero_states=sum(s["positive_actions"] + s["negative_actions"] > 0 for s in state_rows),
        all_edge_measured_positive_actions=sum(s["positive_actions"] for s in state_rows),
        all_edge_best_action_mean_pp=float(np.mean([s["best_action_value_pp"] for s in state_rows])),
        groups=group_summary, saved_r1_decision_matches=decision_count,
        integrity_checks=dict(saved_branch_routes_revalued=revalued_branch_routes,
            matched_endpoint_routes_revalued=revalued_endpoint_routes,
            max_route_cost_error=max_route_error, matched_endpoint_work_caps_checked=True),
        arithmetic_expressibility=dict(positive_action_cases=len(witnesses),
            positive_actions_with_longer_same_penalty_competitor=sum(r["longer_edges_with_same_penalty"] > 0 for r in witnesses),
            state_specific_hindsight_witnesses_selecting_target=sum(r["hindsight_witness_selects_target"] for r in witnesses),
            interpretation="Label-informed constants target each observed edge separately. This proves per-case expressibility only, not a shared generalizable rule, discovery success or deployment benefit."),
        same_instance_endpoint_link=dict(programs=len(paired), original_instances=len(instance_ids),
            program_instance_pairs=len(matched), endpoint_solver_seeds=[0, 1], bank_trajectory_seed=17,
            nonzero_local_program_instance_pairs=sum(r["any_nonzero_local"] for r in matched),
            programs_with_all_zero_local=sum(r["nonzero_states"] == 0 for r in paired),
            zero_local_but_nonzero_full_rule_pairs=sum(not r["any_nonzero_local"] and abs(r["full_rule_benefit_gap_pp"]) > EPS for r in matched),
            zero_local_but_nonzero_full_rule_original_instances=len({r["instance_id"] for r in matched if not r["any_nonzero_local"] and abs(r["full_rule_benefit_gap_pp"]) > EPS}),
            zero_local_but_nonzero_full_rule_programs=len({r["program_sha256"] for r in matched if not r["any_nonzero_local"] and abs(r["full_rule_benefit_gap_pp"]) > EPS}),
            mean_local_program_value_range_pp=[min(localx), max(localx)],
            mean_full_rule_program_benefit_range_gap_pp=[min(fully), max(fully)],
            descriptive_spearman=rank_correlation(localx, fully)),
        limitations=["Eight original instances; states/programs/pairs are not independent samples.",
            "Retrospective, reused discovery data; no generalization or causal attribution claim.",
            "Local branch horizon is not aligned to whole-rule remaining budget.",
            "Bank trajectory seed17 differs from stored endpoint seeds0/1.",
            "Per-state best action/rule uses hindsight labels; not a deployable policy or a full-route gain.",
            "Grammar coverage concerns a fixed finite 144-program set, not all allowed arithmetic expressions.",
            "State-specific existence witnesses are constructed after observing positive labels and are not generalization evidence.",
            "Grammar endpoint results from other original instances are not pooled into matched analysis."],
        resources=dict(wall_seconds=time.perf_counter()-start, process_cpu_seconds=time.process_time()-cpu,
                       python=sys.version, numpy=np.__version__, platform=platform.platform(),
                       new_solver_runs=0, new_branches=0, llm_calls=0, gpu_used=False, final_opened=False))
    write_csv(args.output / "states.csv", state_rows)
    write_csv(args.output / "positive_action_witnesses.csv", witnesses)
    write_csv(args.output / "programs.csv", program_rows)
    write_csv(args.output / "matched_existing_endpoints.csv", matched)
    source = [Path(__file__), CODE / "src/tsp_value/programs.py", CODE / "src/tsp_value/solver.py", CODE / "src/tsp_value/data.py"]
    inputs.update(source)
    for p in source:
        target = args.output / "executed_source" / p.relative_to(CODE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    hashes = {p.relative_to(CODE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
    (args.output / "input_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
