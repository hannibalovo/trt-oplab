#!/usr/bin/env python3
"""Offline TensorRT 10-style LayerInfo summary and conservative structural diff.

No TensorRT or GPU required. A mismatched MHA key is UNRESOLVED, not proof of
loss of fusion, because rewritten graphs may rename or remove ONNX source nodes.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ONNX_NODE = re.compile(r'\[ONNX Layer:\s*([^\]]+)\]')
SDPA = re.compile(r'^node_scaled_dot_product_attention(?:_\d+)?$')


def load(path):
    p = Path(path)
    data = p.read_text(encoding='utf-8')
    try:
        doc = json.loads(data)
    except json.JSONDecodeError as original:
        # Compatibility repair ONLY for observed fixture: a solitary spurious '{'
        # before a complete JSON object. Never guess-repair arbitrary corruption.
        if re.match(r'^\s*\{\s*\{\s*"Layers"\s*:', data):
            candidate = data[data.index('{', data.index('{') + 1):]
            try:
                doc = json.loads(candidate)
                print(f'NOTICE: ignored one spurious leading brace in {p}', file=sys.stderr)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{p}: invalid JSON even after known leading-brace fix: {exc}') from exc
        else:
            raise ValueError(f'{p}: invalid JSON: {original}') from original
    if not isinstance(doc, dict) or not isinstance(doc.get('Layers'), list):
        raise ValueError(f'{p}: expected object with a Layers array (TRT 10-style Inspector JSON)')
    for i, layer in enumerate(doc['Layers']):
        if not isinstance(layer, dict) or not isinstance(layer.get('Name'), str):
            raise ValueError(f'{p}: malformed Layers[{i}]')
    return doc


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nodes(layer):
    return list(dict.fromkeys(ONNX_NODE.findall(layer.get('Metadata', '') or '')))


def shape_list(layer, field):
    return [t.get('Dimensions', None) for t in layer.get(field, []) if isinstance(t, dict)]


def sdpa_key(layer):
    names = [name for name in nodes(layer) if SDPA.fullmatch(name)]
    return names[0] if len(names) == 1 else None


def is_mha(layer):
    # Intentionally strict for this supplied Inspector style; plugins may need
    # custom matching. Do not classify any node merely mentioning SDPA as MHA.
    return '_gemm_mha' in layer.get('Name', '').lower()


def mha_records(doc):
    records = []
    for idx, l in enumerate(doc['Layers']):
        if not is_mha(l):
            continue
        input_shapes = shape_list(l, 'Inputs')
        # Explicit dynamic KV denotes self-attention in the supplied LingBot
        # Action engine; this is a heuristic, not a general Transformer rule.
        dynamic = any(-1 in shape for shape in input_shapes if isinstance(shape, list))
        records.append({
            'inspector_index': idx,
            'name': l['Name'],
            'source_anchor': sdpa_key(l),
            'heuristic_kind': 'self_dynamic_kv' if dynamic else 'cross_fixed_kv',
            'tactic': l.get('TacticName', ''),
            'layer_type': l.get('LayerType', ''),
            'inputs': input_shapes,
            'outputs': shape_list(l, 'Outputs'),
        })
    return records


def summarize(path):
    doc = load(path)
    layers = doc['Layers']
    m = mha_records(doc)
    counts = Counter(l.get('LayerType', '<missing>') for l in layers)
    return {
        'path': str(Path(path)),
        'sha256': digest(path),
        'layer_count': len(layers),
        'layer_type_counts': dict(sorted(counts.items())),
        'mha_count': len(m),
        'mha_kind_counts_HEURISTIC': dict(Counter(r['heuristic_kind'] for r in m)),
        'mha_tactic_counts': dict(Counter(r['tactic'] for r in m)),
        'mha_layers': m,
        'warnings': [
            'Structural inspector only: no Q/K/V numerical data or runtime kernel durations.',
            'Inspector Layer != CUDA kernel launch; kgen/gemm reflect compiler layers.',
            'MHA kind uses only dynamic (-1) KV shape as a LingBot-specific heuristic.',
            'TacticName equality does not prove equal runtime latency or exactly equal GPU launches.',
        ],
    }


def source_ownership(doc):
    owners = defaultdict(set)
    for idx, l in enumerate(doc['Layers']):
        for name in nodes(l):
            owners[name].add(idx)
    return owners


def diff(base_path, cand_path):
    b = load(base_path); c = load(cand_path)
    b_m = mha_records(b); c_m = mha_records(c)
    def unique(records):
        groups = defaultdict(list)
        for row in records:
            if row['source_anchor']:
                groups[row['source_anchor']].append(row)
        return {k: v[0] for k, v in groups.items() if len(v) == 1}
    b_map = unique(b_m); c_map = unique(c_m)
    match = sorted(b_map.keys() & c_map.keys())
    changed = []
    for key in match:
        old = b_map[key]; new = c_map[key]
        deltas = [f for f in ('tactic', 'layer_type', 'inputs', 'outputs') if old[f] != new[f]]
        if deltas:
            changed.append({
                'source_anchor': key,
                'changed_fields': deltas,
                'baseline': old,
                'candidate': new,
            })
    # Compare ownership of unique original ONNX node IDs. May be impossible
    # after a rewrite; report only conservatively, never auto-align by index.
    bo = source_ownership(b); co = source_ownership(c)
    matched_unique_nodes = [n for n in bo.keys() & co.keys() if len(bo[n]) == len(co[n]) == 1]
    ownership_changed = []
    for n in sorted(matched_unique_nodes):
        bi = next(iter(bo[n])); ci = next(iter(co[n]))
        bl = b['Layers'][bi]; cl = c['Layers'][ci]
        if (bl.get('TacticName', '') != cl.get('TacticName', '') or
                bl.get('LayerType', '') != cl.get('LayerType', '') or
                set(nodes(bl)) != set(nodes(cl))):
            ownership_changed.append({
                'onnx_node': n,
                'baseline_index': bi,
                'candidate_index': ci,
                'baseline_layer': bl.get('Name'),
                'candidate_layer': cl.get('Name'),
                'tactic_changed': bl.get('TacticName') != cl.get('TacticName'),
                'fusion_membership_changed': set(nodes(bl)) != set(nodes(cl)),
            })
    return {
        'baseline': {'path': str(base_path), 'sha256': digest(base_path),
                     'layers': len(b['Layers']), 'mha': len(b_m),
                     'types': dict(Counter(x.get('LayerType') for x in b['Layers']))},
        'candidate': {'path': str(cand_path), 'sha256': digest(cand_path),
                      'layers': len(c['Layers']), 'mha': len(c_m),
                      'types': dict(Counter(x.get('LayerType') for x in c['Layers']))},
        'matched_mha_anchors': len(match),
        'changed_mha_count': len(changed),
        'changed_mha': changed,
        'unmatched_baseline_mha_anchors_UNRESOLVED': sorted(b_map.keys()-c_map.keys()),
        'unmatched_candidate_mha_anchors_UNRESOLVED': sorted(c_map.keys()-b_map.keys()),
        'unanchored_baseline_mha': len(b_m)-len(b_map),
        'unanchored_candidate_mha': len(c_m)-len(c_map),
        'matched_unique_onnx_nodes': len(matched_unique_nodes),
        'changed_unique_onnx_node_count': len(ownership_changed),
        'changed_unique_onnx_nodes_first_50': ownership_changed[:50],
        'warnings': [
            'Conservative semantic matching, not index-based alignment.',
            'Missing SDPA anchor may mean renamed Plugin/fusion, NOT a confirmed missing computation.',
            'Unique-node ownership differences are hypotheses to verify in ONNX graph and Nsys.',
            'No timing, quality, or GPU kernel identity can be inferred from this JSON diff.',
        ],
    }


def md_summary(s):
    lines = [f"# Inspector structural summary\n", f"- Source SHA256: `{s['sha256']}`",
             f"- Layers: **{s['layer_count']}**; MHA-labeled: **{s['mha_count']}**",
             f"- Layer types: {s['layer_type_counts']}",
             '', '| Inspector # | Source SDPA anchor | Heuristic kind | Tactic |',
             '|---:|---|---|---|']
    for r in s['mha_layers']:
        lines.append(f"| {r['inspector_index']} | `{r['source_anchor'] or 'UNRESOLVED'}` | "
                     f"{r['heuristic_kind']} | `{r['tactic']}` |")
    lines.extend(['', '## Limits', *(f'- {x}' for x in s['warnings'])])
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    one = sub.add_parser('summary', help='summarize baseline or candidate Engine Inspector')
    one.add_argument('layerinfo')
    one.add_argument('--out', type=Path)
    one.add_argument('--markdown', type=Path, help='optional human-readable MHA summary')
    two = sub.add_parser('diff', help='semantic comparison; never relies on Inspector layer index')
    two.add_argument('baseline'); two.add_argument('candidate'); two.add_argument('--out', type=Path)
    args = p.parse_args()
    try:
        result = summarize(args.layerinfo) if args.cmd == 'summary' else diff(args.baseline, args.candidate)
        encoded = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(encoded, encoding='utf-8')
        else:
            print(encoded)
        if args.cmd == 'summary' and args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(md_summary(result), encoding='utf-8')
    except (OSError, ValueError) as e:
        p.exit(2, f'ERROR: {e}\n')

if __name__ == '__main__':
    main()
