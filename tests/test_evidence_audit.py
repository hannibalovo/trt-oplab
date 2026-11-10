import importlib.util
from pathlib import Path
import json
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('evidence_audit',ROOT/'helpers'/'evidence_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def good_local(device):
    return dict(scope='one reduced attention example; no engine or full-model replacement',
                passed=True, environment={'device':device},
                candidate={'sha256':'a'*64},criteria={'atol':0.02,'rtol':0.02},
                candidate_vs_trt={'passed':True,'finite':True,'outside_tolerance':0,'elements':4096,'mae':0.001,'max_abs':0.002},
                stored_fp32_reference_reproduced={'passed':True,'finite':True,'outside_tolerance':0},
                behavior={k:True for k in audit.GUARDS},repeated_run_behavior={k:True for k in audit.GUARDS},
                repeated_output_bitwise_equal=True,
                perturbations={k:dict(passed=True,reference_changed=True,output_changed=True,
                                      vs_fp32={'passed':True}) for k in ('q','k','v')})


class EvidenceAuditTests(unittest.TestCase):
    def test_positive_and_scope(self):
        with tempfile.TemporaryDirectory() as td:
            c=Path(td)/'cpu.json';g=Path(td)/'cuda.json'
            c.write_text(json.dumps(good_local('cpu')));g.write_text(json.dumps(good_local('cuda')))
            result=audit.assess(cpu=c,cuda=g)
            self.assertTrue(result['report_checks_passed'])
            self.assertFalse(result['independently_recomputed'])
            self.assertTrue(any('No safetensors' in x for x in result['limitations']))
    def test_faults_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            c=Path(td)/'cpu.json';v=good_local('cpu')
            v['perturbations']['k']['output_changed']=False
            v['behavior']['output_guards_unchanged']=False
            c.write_text(json.dumps(v))
            result=audit.assess(cpu=c)
            self.assertFalse(result['report_checks_passed'])
            self.assertGreaterEqual(len(result['issues']),2)
    def test_capture_not_internal_proof(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cap.json'
            path.write_text(json.dumps(dict(capture_passed=True,all_layer_tactics_identical=True,
                original_layer_count=643,debug_layer_count=643,
                external_output_metrics={'out':{'bitwise_equal':True,'finite':True}})))
            result=audit.assess(capture=path)
            self.assertTrue(result['report_checks_passed'])
            self.assertTrue(any('not independent proof' in x for x in result['limitations']))

if __name__=='__main__':
    unittest.main()
