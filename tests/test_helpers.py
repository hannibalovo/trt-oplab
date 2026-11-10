import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

ROOT = Path(__file__).resolve().parents[1]

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'helpers' / (name+'.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

layerinfo = module('layerinfo')
experiment = module('experiment')

class LayerInfoTests(unittest.TestCase):
    def test_parse_and_diff(self):
        example = {
          'Layers': [
            {'Name':'_gemm_mha_v2_0','LayerType':'kgen','TacticName':'a',
             'Metadata':'[ONNX Layer: node_scaled_dot_product_attention]',
             'Inputs':[{'Dimensions':[48,32,128]},{'Dimensions':[48,128,-1]}],
             'Outputs':[{'Dimensions':[48,32,128]}]},
            {'Name':'ffn','LayerType':'gemm','TacticName':'x',
             'Metadata':'[ONNX Layer: node_linear_10]'},
          ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base.json'; cand=Path(tmp)/'cand.json'
            base.write_text(json.dumps(example))
            modified=json.loads(json.dumps(example))
            modified['Layers'][0]['TacticName']='b'
            cand.write_text(json.dumps(modified))
            s=layerinfo.summarize(base)
            self.assertEqual(s['layer_count'],2)
            self.assertEqual(s['mha_count'],1)
            d=layerinfo.diff(base,cand)
            self.assertEqual(d['changed_mha_count'],1)
            self.assertEqual(d['matched_mha_anchors'],1)
    def test_spurious_brace_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';p.write_text('{\n{"Layers": []}')
            self.assertEqual(len(layerinfo.load(p)['Layers']),0)
            p.write_text('{not json!')
            with self.assertRaises(ValueError): layerinfo.load(p)

class ExperimentTests(unittest.TestCase):
    def test_stage_evidence_checks_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';experiment.init(root,'test faster','action MHA')
            state=experiment.read(root)
            self.assertFalse(experiment.issues(root,state))
            state['stages']['baseline'].update(status='pass',note='measured',evidence=['missing.json'])
            self.assertTrue(experiment.issues(root,state))
            (root/'test.json').write_text('{}')
            state['stages']['baseline']['evidence']=['test.json']
            state['workload_contract'].update({key:'declared' for key in ('device','trt_version','shape_and_kv_state','precision','model_and_engine_hashes','primary_metric_and_acceptance_rule')})
            self.assertFalse(experiment.issues(root,state))
            state['stages']['local'].update(status='pass',note='tested',evidence=['test.json'])
            self.assertTrue(experiment.issues(root,state))
    def test_bad_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';experiment.init(root,'hyp','target')
            (root/'a.json').write_text('{}')
            s=experiment.read(root)
            s['workload_contract'].update({key:'declared' for key in ('device','trt_version','shape_and_kv_state','precision','model_and_engine_hashes','primary_metric_and_acceptance_rule')})
            s['study_boundary']='post-RoPE QKV to attention O'

            for stage in ('baseline','capture'):
                s['stages'][stage].update(status='pass',note='ok',evidence=['a.json'])
            s['stages']['local'].update(status='pass',note='tested',evidence=['a.json'],metrics={
                'numeric_validation_pass': False,'nan_inf_count':0,
                'candidate_kernel_ms':0.9,'candidate_operator_ms':1.0,
                'benchmark_comparison_status':'unavailable'})
            self.assertTrue(any('numeric acceptance failed' in a for a in experiment.issues(root,s)))

class MixedToleranceGateTests(unittest.TestCase):
    def test_relative_threshold_does_not_compare_scalar_abs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';experiment.init(root,'hyp','target')
            (root/'a.json').write_text('{}')
            s=experiment.read(root)
            s['workload_contract'].update({key:'declared' for key in (
                'device','trt_version','shape_and_kv_state','precision',
                'model_and_engine_hashes','primary_metric_and_acceptance_rule')})
            s['study_boundary']='post-RoPE attention'
            for stage in ('baseline','capture'):
                s['stages'][stage].update(status='pass',note='ok',evidence=['a.json'])
            s['stages']['local'].update(status='pass',note='report checked, timing measured',evidence=['a.json'],metrics={
                'numeric_validation_pass':True,'nan_inf_count':0,
                'max_abs_error':0.03,'numeric_tolerance':0.02,
                'candidate_kernel_ms':0.9,'candidate_operator_ms':1.0,
                'benchmark_comparison_status':'unavailable'})
            # A larger max abs can be legal under atol+rtol; stage checker
            # trusts explicit per-element validation from the project harness.
            self.assertFalse(experiment.issues(root,s))

if __name__=='__main__':
    unittest.main()
