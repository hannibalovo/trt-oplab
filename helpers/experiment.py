#!/usr/bin/env python3
"""Evidence-oriented experiment state. Verifies record consistency, NOT GPU correctness.

No external packages. All evidence paths must be existing regular files.
Use `status` as a reproducibility audit, never as a model validation oracle.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

STAGES = ('baseline', 'capture', 'local', 'integration', 'e2e')
STATUSES = ('pending', 'running', 'pass', 'fail', 'blocked', 'skip')
REQUIRED_METRICS = {
    'local': ('numeric_validation_pass', 'nan_inf_count',
              'candidate_kernel_ms', 'candidate_operator_ms',
              'benchmark_comparison_status'),
    'integration': ('numeric_validation_pass', 'nan_inf_count',
                    'baseline_engine_ms', 'candidate_engine_ms'),
    'e2e': ('baseline_chunk_wall_ms', 'candidate_chunk_wall_ms', 'quality_guardrail_pass'),
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def read(root):
    p = root / 'experiment.json'
    if not p.is_file():
        raise ValueError(f'Missing experiment state: {p}')
    return json.loads(p.read_text(encoding='utf-8'))


def write(root, state):
    path = root / 'experiment.json'
    tmp = root / 'experiment.json.tmp'
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def event(root, action, payload):
    with (root / 'history.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'at': now(), 'action': action, **payload}, ensure_ascii=False) + '\n')


def init(root, hypothesis, target):
    if root.exists():
        raise ValueError(f'Output directory already exists: {root} (never overwrite)')
    root.mkdir(parents=True)
    state = {
        'schema_version': 2,
        'id': root.name,
        'created_utc': now(),
        'hypothesis': hypothesis.strip(),
        'target': target.strip(),
        'workload_contract': {
            'device': '', 'trt_version': '', 'cuda_version': '',
            'shape_and_kv_state': '', 'precision': '',
            'model_and_engine_hashes': '',
            'primary_metric_and_acceptance_rule': ''},
        'study_boundary': '',
        'candidate_integration_boundary': '',
        'stages': {name: {'status': 'pending', 'note': '', 'evidence': [], 'metrics': {}}
                   for name in STAGES},
        'decision': {'status': 'undecided', 'note': '', 'evidence': []},
    }
    if not state['hypothesis'] or not state['target']:
        raise ValueError('A non-empty hypothesis and target are required')
    write(root, state)
    event(root, 'init', {'hypothesis': hypothesis, 'target': target})
    return state


def parse_metric(s):
    if '=' not in s:
        raise ValueError(f'Expected metric KEY=VALUE, got {s!r}')
    k, v = s.split('=', 1)
    if not k.strip():
        raise ValueError('Empty metric key')
    if v.lower() in ('true', 'false'):
        val = v.lower() == 'true'
    else:
        try:
            val = float(v)
            if not (val == val and abs(val) != float('inf')):
                raise ValueError(f'Nonfinite metric: {s}')
        except ValueError:
            # Accept strings for nonnumeric metadata, but numeric gate fields will
            # be rejected by validation later.
            if v.lower() in ('nan', 'inf', '-inf', '+inf'):
                raise ValueError(f'Nonfinite metric: {s}')
            val = v
    return k.strip(), val


def issues(root, state):
    out = []
    passed = state.get('stages', {}).get('baseline', {}).get('status') == 'pass'
    if passed:
        contract = state.get('workload_contract', {})
        for key in ('device', 'trt_version', 'shape_and_kv_state', 'precision',
                    'model_and_engine_hashes', 'primary_metric_and_acceptance_rule'):
            if not str(contract.get(key, '')).strip():
                out.append(f'baseline: missing workload_contract.{key}')
    if state.get('stages', {}).get('capture', {}).get('status') == 'pass':
        if not str(state.get('study_boundary', '')).strip():
            out.append('capture: study_boundary must be defined')
    if state.get('stages', {}).get('integration', {}).get('status') == 'pass':
        if not str(state.get('candidate_integration_boundary', '')).strip():
            out.append('integration: candidate_integration_boundary must be defined')
    for name in STAGES:
        stage = state.get('stages', {}).get(name, {})
        status = stage.get('status')
        if status not in STATUSES:
            out.append(f'{name}: invalid status {status!r}')
            continue
        if status != 'pass':
            continue
        if not stage.get('note', '').strip():
            out.append(f'{name}: pass requires explanatory note')
        evidence = stage.get('evidence', [])
        if not evidence:
            out.append(f'{name}: pass requires at least one evidence file')
        for s in evidence:
            if not isinstance(s, str):
                out.append(f'{name}: evidence path must be a string')
                continue
            p = Path(s)
            p = p if p.is_absolute() else root / p
            if not p.is_file():
                out.append(f'{name}: missing evidence file: {p}')
        metrics = stage.get('metrics', {})
        for key in REQUIRED_METRICS.get(name, ()):
            if key not in metrics:
                out.append(f'{name}: missing metric {key}')
        if name in ('local', 'integration'):
            # Per-element mixed atol+rtol acceptance cannot be inferred from
            # max_abs_error alone. The reference validator must report a pass.
            if metrics.get('numeric_validation_pass') is not True:
                out.append(f'{name}: numeric acceptance failed or missing explicit numeric_validation_pass=true')
            if type(metrics.get('nan_inf_count')) not in (float, int) or metrics['nan_inf_count'] != 0:
                out.append(f'{name}: numeric acceptance failed: nonzero/invalid nan_inf_count')
        if name == 'local':
            comparison = metrics.get('benchmark_comparison_status')
            if comparison not in ('matched', 'unavailable'):
                out.append('local: benchmark_comparison_status must be matched or unavailable')
            if comparison == 'matched':
                if type(metrics.get('baseline_operator_ms')) not in (float, int):
                    out.append('local: matched comparison requires baseline_operator_ms')
                if metrics.get('workload_boundary_matched') is not True:
                    out.append('local: matched comparison requires workload_boundary_matched=true')
            # unavailable is a valid honest local checkpoint, NOT a speedup claim.
        if name == 'e2e' and metrics.get('quality_guardrail_pass') is not True:
            out.append('e2e: quality_guardrail_pass must be true for pass')
        for k in ('baseline_kernel_ms', 'candidate_kernel_ms',
                  'baseline_operator_ms', 'candidate_operator_ms',
                  'baseline_chunk_wall_ms', 'candidate_chunk_wall_ms',
                  'baseline_engine_ms', 'candidate_engine_ms'):
            if k in metrics and (type(metrics[k]) not in (int, float) or metrics[k] <= 0):
                out.append(f'{name}: {k} must be nonnegative numeric')
    for i, name in enumerate(STAGES):
        if state['stages'][name]['status'] == 'pass':
            unpassed = [x for x in STAGES[:i] if state['stages'][x]['status'] != 'pass']
            if unpassed:
                out.append(f'{name}: preceding stages not passed: {unpassed}')
    if state.get('decision', {}).get('status') == 'accept':
        if state['stages']['e2e']['status'] != 'pass':
            out.append('accept decision requires passed e2e evidence')
        else:
            m = state['stages']['e2e']['metrics']
            if 'baseline_chunk_wall_ms' in m and 'candidate_chunk_wall_ms' in m:
                if m['candidate_chunk_wall_ms'] >= m['baseline_chunk_wall_ms']:
                    out.append('accept decision requires end-to-end latency improvement (or use iterate/reject)')
        if not state['decision'].get('note', '').strip():
            out.append('accept decision requires an explanation')
    return out


def status(root):
    st = read(root)
    print(f"Experiment: {st['id']} — {st['hypothesis']}")
    for name in STAGES:
        stage = st['stages'][name]
        print(f"  {name:11s} {stage['status']:8s} evidence={len(stage['evidence'])}")
    print(f"Decision: {st['decision']['status']}")
    probs = issues(root, st)
    print(f'Evidence validation: {"OK" if not probs else "ISSUES"}')
    for p in probs:
        print(f'  ! {p}')
    next_stage = next((n for n in STAGES if st['stages'][n]['status'] != 'pass'), None)
    print(f'Next unchecked stage: {next_stage or "none"}')
    return not probs


def set_stage(root, args):
    st = read(root)
    stage = st['stages'][args.name]
    before = json.loads(json.dumps(stage))
    if args.status == 'pass' and not args.note:
        raise ValueError('pass requires --note with actual verification outcome')
    if args.status == 'pass' and not (stage['evidence'] or args.evidence):
        raise ValueError('pass requires --evidence existing path')
    stage['status'] = args.status
    if args.note is not None:
        stage['note'] = args.note
    stage['evidence'] = list(dict.fromkeys(stage['evidence'] + (args.evidence or [])))
    stage['metrics'].update(dict(parse_metric(s) for s in (args.metric or [])))
    errors = issues(root, st)
    if args.status == 'pass' and errors:
        raise ValueError('Cannot mark pass:\n- ' + '\n- '.join(errors))
    write(root, st)
    event(root, 'stage', {'stage': args.name, 'old': before, 'new': stage})
    print(f'{args.name}: {before["status"]} → {args.status}')


def set_decision(root, args):
    st = read(root)
    st['decision'] = {'status': args.status, 'note': args.note, 'evidence': args.evidence or []}
    errors = issues(root, st)
    if args.status == 'accept' and errors:
        raise ValueError('Cannot accept:\n- ' + '\n- '.join(errors))
    write(root, st)
    event(root, 'decision', st['decision'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('init')
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--hypothesis', required=True)
    a.add_argument('--target', required=True)
    b = sub.add_parser('status'); b.add_argument('root', type=Path)
    c = sub.add_parser('stage'); c.add_argument('root', type=Path)
    c.add_argument('--name', choices=STAGES, required=True)
    c.add_argument('--status', choices=STATUSES, required=True)
    c.add_argument('--evidence', action='append')
    c.add_argument('--metric', action='append')
    c.add_argument('--note')
    d = sub.add_parser('decision'); d.add_argument('root', type=Path)
    d.add_argument('--status', choices=('accept', 'iterate', 'reject', 'blocked'), required=True)
    d.add_argument('--note', required=True)
    d.add_argument('--evidence', action='append')
    args = p.parse_args()
    try:
        if args.cmd == 'init':
            init(args.out, args.hypothesis, args.target)
            print(f'Created {args.out / "experiment.json"}')
        elif args.cmd == 'status':
            return 0 if status(args.root) else 2
        elif args.cmd == 'stage':
            set_stage(args.root, args)
        else:
            set_decision(args.root, args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        p.exit(2, f'ERROR: {e}\n')
    return 0

if __name__ == '__main__':
    sys.exit(main())
