"""Auditable synthetic biology checks. Python 3.10+, standard library only.

This is an executable experiment, not a medical dose calculator. All compounds,
exposure limits, coverage cutoffs and patient profiles below are invented.
Run: python demo.py --output results/simulations.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from statistics import mean


@dataclass(frozen=True)
class Decision:
    stage: str
    status: str
    reason: str


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def pk_check(record):
    """Check a VERIFIED protocol record, never fields asserted by the LLM.

    HED uses the adult-reference FDA Km convention (mouse 3; adult human 37).
    Exposure formula is valid ONLY for a single, instantaneous IV bolus in the
    specified linear one-compartment model. No oral/repeated-dose inference.
    """
    keys = ('mouse_noael_mg_kg', 'safety_factor', 'proposed_mg_kg',
            'human_weight_kg', 'volume_l', 'cmax_limit_mg_l')
    if any(not number(record.get(k)) or record[k] <= 0 for k in keys):
        return Decision('exposure', 'abstain', 'Missing or invalid exposure inputs')
    if record.get('pk_model') != 'single_iv_bolus_linear_one_compartment':
        return Decision('exposure', 'abstain', 'Exposure model does not cover this route or schedule')
    if record['safety_factor'] < 1:
        return Decision('exposure', 'abstain', 'Invalid safety factor')
    ceiling = record['mouse_noael_mg_kg'] * 3 / 37 / record['safety_factor']
    cmax = record['proposed_mg_kg'] * record['human_weight_kg'] / record['volume_l']
    if record['proposed_mg_kg'] > ceiling + 1e-12:
        return Decision('exposure', 'reject', 'Exceeds declared HED / safety-factor ceiling')
    if cmax > record['cmax_limit_mg_l']:
        return Decision('exposure', 'reject', 'Exceeds declared Cmax ceiling')
    return Decision('exposure', 'pass', 'Within the two declared toy bounds')


def dag_check(record):
    """Total-effect adjustment check using the back-door criterion.

    Reject exposure descendants as adjustment variables. Delete arrows leaving
    the exposure, take ancestors of exposure/outcome/adjustment, moralize that
    graph, and test separation after deleting adjustment nodes. This tests a
    supplied DAG; it cannot establish the DAG from observational measurements.
    """
    try:
        edges = record['edges']
        x, y = record['exposure'], record['outcome']
        adjustment = set(record['adjustment'])
        observed = set(record['observed'])
        if not isinstance(x, str) or not isinstance(y, str) or x == y:
            raise ValueError('invalid target')
        if not isinstance(edges, list) or any(not isinstance(e, list) or len(e) != 2 or
                any(not isinstance(n, str) or not n for n in e) for e in edges):
            raise ValueError('invalid edges')
        nodes = {n for edge in edges for n in edge}
        if not {x, y} <= nodes or not adjustment <= nodes or {x, y} & adjustment:
            raise ValueError('invalid target or adjustment')
        children = {n: set() for n in nodes}
        parents = {n: set() for n in nodes}
        for a, b in edges:
            children[a].add(b)
            parents[b].add(a)
        indegree = {n: len(parents[n]) for n in nodes}
        ready = [n for n in nodes if indegree[n] == 0]
        visited = 0
        while ready:
            n = ready.pop()
            visited += 1
            for child in children[n]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(nodes):
            return Decision('causality', 'abstain', 'Graph contains a directed cycle')
        if not adjustment <= observed:
            return Decision('causality', 'abstain', 'Adjustment requires an unmeasured variable')
        descendants, stack = set(), list(children[x])
        while stack:
            n = stack.pop()
            if n not in descendants:
                descendants.add(n)
                stack.extend(children[n])
        if descendants & adjustment:
            return Decision('causality', 'reject', 'Total-effect adjustment includes an exposure descendant')
        back_edges = [(a, b) for a, b in edges if a != x]
        back_parents = {n: set() for n in nodes}
        for a, b in back_edges:
            back_parents[b].add(a)
        ancestors, stack = set(), [x, y, *adjustment]
        while stack:
            n = stack.pop()
            if n not in ancestors:
                ancestors.add(n)
                stack.extend(back_parents[n])
        moral = {n: set() for n in ancestors}
        for child in ancestors:
            ps = back_parents[child] & ancestors
            for p in ps:
                moral[p].add(child)
                moral[child].add(p)
                for other in ps - {p}:
                    moral[p].add(other)
        seen, stack = set(), [x]
        while stack:
            n = stack.pop()
            if n in seen or n in adjustment:
                continue
            seen.add(n)
            stack.extend(moral[n] - seen - adjustment)
        if y in seen:
            return Decision('causality', 'reject', 'Unblocked back-door path in the declared DAG')
        return Decision('causality', 'pass', 'Adjustment satisfies the declared back-door criterion')
    except (KeyError, TypeError, ValueError):
        return Decision('causality', 'abstain', 'Missing or invalid causal specification')


def sensor_check(record):
    """A predeclared coverage rule, NOT a calibrated probability of truth."""
    critical = ('renal_function', 'liver_function', 'sampling_time')
    if any(record.get(k) is not True for k in critical):
        return Decision('coverage', 'abstain', 'Insufficient sensor data')
    fraction = record.get('observed_fraction')
    if not number(fraction) or not 0 <= fraction <= 1 or fraction < 0.8:
        return Decision('coverage', 'abstain', 'Insufficient sensor data')
    if record.get('assay_in_distribution') is not True:
        return Decision('coverage', 'abstain', 'Insufficient sensor data')
    return Decision('coverage', 'pass', 'Critical measurements present; declared coverage >= 80%')


def pipeline(case):
    decisions = []
    for key, check in [('pk', pk_check), ('dag', dag_check), ('sensors', sensor_check)]:
        result = check(case.get(key, {}))
        decisions.append(asdict(result))
        if result.status != 'pass':
            break
    return decisions


def fixture_cases():
    pk = dict(mouse_noael_mg_kg=10, safety_factor=10, proposed_mg_kg=0.05,
              human_weight_kg=70, volume_l=35, cmax_limit_mg_l=0.15,
              pk_model='single_iv_bolus_linear_one_compartment')
    dag = dict(edges=[['illness', 'cholesterol'], ['illness', 'mortality'],
                      ['cholesterol', 'mediator'], ['mediator', 'mortality']],
               exposure='cholesterol', outcome='mortality', adjustment=['illness'],
               observed=['cholesterol', 'mortality', 'illness', 'mediator'])
    sensors = dict(renal_function=True, liver_function=True, sampling_time=True,
                   observed_fraction=0.9, assay_in_distribution=True)
    def make(name, expected, p=None, d=None, s=None):
        return dict(name=name, expected=expected, pk={**pk, **(p or {})},
                    dag={**dag, **(d or {})}, sensors={**sensors, **(s or {})})
    return [
        make('complete_control', 'pass'),
        make('dose_arithmetic_error', 'exposure', p={'proposed_mg_kg': 0.2}),
        make('exposure_limit_error', 'exposure', p={'volume_l': 10}),
        make('oral_route_without_model', 'exposure', p={'pk_model': 'oral_unknown'}),
        make('unadjusted_common_cause', 'causality', d={'adjustment': []}),
        make('mediator_adjustment', 'causality', d={'adjustment': ['illness', 'mediator']}),
        make('hidden_common_cause', 'causality', d={'observed': ['cholesterol', 'mortality']}),
        make('renal_measurement_missing', 'coverage', s={'renal_function': False}),
        make('low_coverage', 'coverage', s={'observed_fraction': 0.79}),
        make('mouse_assay_shift', 'coverage', s={'assay_in_distribution': False}),
    ]


def selection_simulation(n=100000, seed=15112026, negative_retention=0.2):
    rng = random.Random(seed)
    true_effects, observed_effects = [], []
    for _ in range(n):
        theta = rng.gauss(0.2, 0.3)
        observed = rng.gauss(theta, 0.5)
        if observed > 0 or rng.random() < negative_retention:
            true_effects.append(theta)
            observed_effects.append(observed)
    observed = mean(observed_effects)
    true = mean(true_effects)
    return dict(seed=seed, experiments=n, published=len(true_effects),
                positive_fraction=mean(x > 0 for x in observed_effects),
                mean_reported=observed, mean_true_selected=true,
                correction_fraction=1 - true / observed,
                shrinkage_slope=0.3**2 / (0.3**2 + 0.5**2))


def compression(d=23030, k=300):
    def variance(tau):
        return sum(math.exp(-i/tau) for i in range(k)) / sum(math.exp(-i/tau) for i in range(d))
    return dict(measurements=d, retained_coordinates=k,
                raw_bits=d * 16, token_upper_bits=300 * math.log2(100000),
                isotropic_retained_fraction=k/d, low_rank_retained_fraction=variance(200),
                assumptions='16-bit measurement quantization; 100k token vocabulary; diagonal covariance with eigenvalues exp(-i/200) or equal eigenvalues. Token count is not a variance estimate.')


def confounding():
    # Integer expected counts from a hypothetical 10,000-person population.
    # Disease prevalence 10%; P(low|disease)=.8, P(low|healthy)=.1.
    # Mortality is .4 with disease, .02 otherwise, independent of cholesterol.
    return dict(low_group=1700, low_deaths=338, other_group=8300, other_deaths=242,
                observational_rr=(338/1700)/(242/8300),
                interventional_risk=0.1*0.4+0.9*0.02,
                causal_effect=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cases = []
    for case in fixture_cases():
        trace = pipeline(case)
        actual = 'pass' if all(d['status'] == 'pass' for d in trace) else trace[-1]['stage']
        assert actual == case['expected'], (case['name'], actual)
        cases.append(dict(name=case['name'], expected=case['expected'], trace=trace))
    result = dict(kind='synthetic, not model responses or clinical validation',
                  selection=selection_simulation(), compression=compression(),
                  confounding=confounding(), fixtures=cases)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
