#!/usr/bin/env python3
"""Audit reported capture/local-validation JSON without raw tensors or GPU.

This validates consistency and strength of *claims in supplied reports*. It does
not verify safetensors bytes, recompute outputs, or prove TRT debug equivalence.
Standard library only; no project data is embedded in this helper.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

GUARDS = ('caller_output_preserved', 'inputs_unchanged',
          'output_finite_and_written', 'output_guards_unchanged',
          'output_shape_dtype')

def load(path):
    if path is None:
        return None
    with Path(path).open(encoding='utf-8') as f:
        v = json.load(f)
    if not isinstance(v, dict):
        raise ValueError(f'{path}: expected JSON object')
    return v


def report_issue(issues, where, what):
    issues.append(f'{where}: {what}')


def assess_local(label, data, issues, warnings):
    if data is None:
        return None
    if data.get('scope') != 'one reduced attention example; no engine or full-model replacement':
        warnings.append(f'{label}: different or unrecognized validation scope; review manually')
    device = (data.get('environment') or {}).get('device')
    if device and device != label:
        report_issue(issues, label, f'environment.device={device!r} expected {label!r}')
    direct = data.get('candidate_vs_trt') or {}
    fp32 = data.get('stored_fp32_reference_reproduced') or {}
    if data.get('passed') is not True:
        report_issue(issues, label, 'top-level passed is not true')
    for name, d in (('candidate_vs_trt',direct),('stored_fp32_reference_reproduced',fp32)):
        if d.get('passed') is not True or d.get('finite') is not True or d.get('outside_tolerance') != 0:
            report_issue(issues, label, f'{name} did not report a finite zero-violations PASS')
    if data.get('repeated_output_bitwise_equal') is not True:
        report_issue(issues, label, 'repeat bitwise-equality check not passed')
    for category in ('behavior', 'repeated_run_behavior'):
        checks=data.get(category) or {}
        for k in GUARDS:
            if checks.get(k) is not True:
                report_issue(issues, label, f'{category}.{k} is not true')
    cases = data.get('perturbations') or {}
    for key in ('q','k','v'):
        d=cases.get(key) or {}
        if d.get('passed') is not True or d.get('reference_changed') is not True or d.get('output_changed') is not True:
            report_issue(issues, label, f'perturbation.{key} not passed/consumed')
        if (d.get('vs_fp32') or {}).get('passed') is not True:
            report_issue(issues, label, f'perturbation.{key} reference check not passed')
    criteria=data.get('criteria') or {}
    if not all(isinstance(criteria.get(k), (float,int)) and not isinstance(criteria.get(k), bool)
               for k in ('atol','rtol')):
        report_issue(issues, label, 'missing numeric tolerance')
    if not (data.get('candidate') or {}).get('sha256'):
        warnings.append(f'{label}: candidate code hash absent')
    return {
        'device':device,
        'reported_pass':data.get('passed') is True,
        'candidate_sha256':(data.get('candidate') or {}).get('sha256'),
        'elements':direct.get('elements'),
        'outside_tolerance':direct.get('outside_tolerance'),
        'candidate_vs_trt_mae':direct.get('mae'),
        'candidate_vs_trt_max_abs':direct.get('max_abs'),
        'atol':criteria.get('atol'),'rtol':criteria.get('rtol'),
    }


def assess_capture(data, issues, warnings):
    if data is None:
        return None
    if data.get('capture_passed') is not True:
        report_issue(issues, 'capture', 'capture_passed not true')
    if data.get('all_layer_tactics_identical') is not True:
        report_issue(issues, 'capture', 'reported layer tactics not identical')
    a,b=data.get('original_layer_count'),data.get('debug_layer_count')
    if not (isinstance(a,int) and a>0 and a==b):
        report_issue(issues, 'capture', 'original/debug layer counts missing or unequal')
    outputs=data.get('external_output_metrics') or {}
    if not outputs:
        report_issue(issues, 'capture', 'external output comparisons missing')
    for name, m in outputs.items():
        if m.get('bitwise_equal') is not True or m.get('finite') is not True:
            report_issue(issues, 'capture', f'external output {name} not bitwise equal and finite')
    if not data.get('original_engine_sha256') or not data.get('debug_engine_sha256'):
        warnings.append('capture: engine hashes missing')
    warnings.append('capture: tactic equality and external bitwise equality are REPORTS, not independent proof of identical internal numerical values')
    return {
        'source_run_id':data.get('source_run_id'),
        'layer_name':(data.get('layer_mapping') or {}).get('name'),
        'original_layer_count':a,'debug_layer_count':b,
        'reported_tactics_identical':data.get('all_layer_tactics_identical'),
        'reported_external_outputs_bitwise_equal': sorted(k for k,m in outputs.items() if m.get('bitwise_equal') is True),
    }


def verify_hashes(manifest_path, issues, warnings, *, source_capture=None, contract=None):
    if manifest_path is None:
        return None
    path=Path(manifest_path)
    manifest=load(path)
    declared=manifest.get('files_sha256') or {}
    confirmed=[]; unavailable=[]
    for rel, expected in declared.items():
        override = {'evidence/source_capture.json': source_capture, 'contract.json': contract}.get(rel)
        dest=Path(override) if override is not None else path.parent/rel
        if not dest.is_file():
            unavailable.append(rel)
            continue
        actual=hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual != expected:
            report_issue(issues,'manifest',f'{rel}: SHA256 mismatch')
        else:
            confirmed.append(rel)
    if unavailable:
        warnings.append(f'manifest: cannot check {len(unavailable)} absent files; raw tensor integrity remains unverified')
    return {'confirmed_files':confirmed,'missing_files':unavailable,
            'source_hashes_only_reported':list((manifest.get('source_files_sha256') or {}).keys())}


def assess(*, cpu=None, cuda=None, capture=None, manifest=None, contract=None, candidate=None):
    issues=[]; warnings=[]
    cap=assess_capture(load(capture),issues,warnings) if capture else None
    local={}
    for name,path in (('cpu',cpu),('cuda',cuda)):
        if path:
            local[name]=assess_local(name,load(path),issues,warnings)
    if 'cpu' in local and 'cuda' in local:
        x,y=local['cpu'],local['cuda']
        if x['candidate_sha256']!=y['candidate_sha256']:
            report_issue(issues,'cross_device','different candidate code SHA256')
        for k in ('elements','outside_tolerance','atol','rtol','candidate_vs_trt_mae','candidate_vs_trt_max_abs'):
            if x[k]!=y[k]:
                warnings.append(f'cross_device: {k} differs between CPU and CUDA; review expected numeric behavior')
    cfg=load(contract) if contract is not None else None
    contract_summary=None
    if cfg is not None:
        semantics=cfg.get('semantics') or {}
        contract_summary={'operator':cfg.get('operator'),
                          'formula':semantics.get('formula'),
                          'scale':semantics.get('scale'),
                          'tensor_shapes':{k:(v or {}).get('shape') for k,v in (cfg.get('tensors') or {}).items()}}
        numeric=cfg.get('numeric_check') or {}
        for label,d in local.items():
            if d['atol']!=numeric.get('atol') or d['rtol']!=numeric.get('rtol'):
                report_issue(issues,label,'reported tolerance differs from contract.json')
            expected_elements=1
            for dim in (cfg.get('tensors') or {}).get('o_trt',{}).get('shape',[]):
                expected_elements*=dim
            if expected_elements!=d['elements']:
                report_issue(issues,label,'output element count differs from contract.json')
    if candidate is not None:
        actual=hashlib.sha256(Path(candidate).read_bytes()).hexdigest()
        for label,d in local.items():
            if d['candidate_sha256']!=actual:
                report_issue(issues,label,'candidate source file hash differs from report')
    hashes=verify_hashes(manifest,issues,warnings,source_capture=capture,contract=contract)
    warnings.extend([
        'No safetensors: cannot recompute FP32 or validate the actual captured Q/K/V/O tensors.',
        'No performance evidence: do not claim kernel, operator, Engine or application acceleration.',
        'One reduced fixture is not a full-Engine, multiple-state, original-TRT-kernel or Plugin-integration test.',
    ])
    return {'schema_version':1,'audit_scope':'JSON-report consistency only',
            'report_checks_passed':not issues,'independently_recomputed':False,
            'captured_report':cap,'local_reports':local,'contract_summary':contract_summary,
            'candidate_source_hash_checked': candidate is not None, 'manifest_hash_audit':hashes,
            'issues':issues,'limitations':warnings}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('cpu','cuda','capture','manifest','contract','candidate'):
        p.add_argument('--'+arg,type=Path)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if not any(getattr(args,k) for k in ('cpu','cuda','capture','manifest','contract','candidate')):
        p.error('provide at least one evidence file')
    try:
        result=assess(cpu=args.cpu,cuda=args.cuda,capture=args.capture,manifest=args.manifest,
                      contract=args.contract,candidate=args.candidate)
        data=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        if args.out:
            if args.out.exists():
                p.error('output exists; choose a fresh path to retain previous audit')
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(data,encoding='utf-8')
            print(f'Wrote {args.out}')
        else:
            print(data,end='')
        return 0 if result['report_checks_passed'] else 2
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as e:
        p.exit(2,f'ERROR: {e}\n')

if __name__=='__main__':
    sys.exit(main())
