"""Tests target failure modes, including back-door collider/descendant mistakes."""
import copy
import math
import unittest
from demo import compression, dag_check, fixture_cases, pipeline, pk_check, sensor_check


class Checks(unittest.TestCase):
    def test_known_failures_stop_at_expected_stage(self):
        for case in fixture_cases():
            with self.subTest(case=case['name']):
                trace = pipeline(case)
                actual = 'pass' if all(x['status']=='pass' for x in trace) else trace[-1]['stage']
                self.assertEqual(actual, case['expected'])

    def test_bad_exposure_values_abstain(self):
        for bad in [None, float('nan'), float('inf'), -1, True, '10']:
            record = copy.deepcopy(fixture_cases()[0]['pk'])
            record['volume_l'] = bad
            self.assertEqual(pk_check(record).status, 'abstain')

    def test_collider_adjustment_opens_backdoor(self):
        graph = dict(edges=[['a','x'],['a','c'],['b','c'],['b','y']],
                     exposure='x',outcome='y',observed=['x','y','c'],adjustment=[])
        self.assertEqual(dag_check(graph).status,'pass')
        graph['adjustment']=['c']
        self.assertEqual(dag_check(graph).status,'reject')

    def test_feedback_graph_is_not_a_dag(self):
        graph=copy.deepcopy(fixture_cases()[0]['dag'])
        graph['edges'].append(['mortality','illness'])
        self.assertEqual(dag_check(graph).status,'abstain')

    def test_exact_sensor_boundary_and_critical_override(self):
        record=copy.deepcopy(fixture_cases()[0]['sensors'])
        record['observed_fraction']=.8
        self.assertEqual(sensor_check(record).status,'pass')
        record['liver_function']=False
        self.assertEqual(sensor_check(record).reason,'Insufficient sensor data')

    def test_compression_full_rank_retains_everything(self):
        r=compression(d=300,k=300)
        self.assertEqual(r['isotropic_retained_fraction'],1)
        self.assertTrue(math.isclose(r['low_rank_retained_fraction'],1))


if __name__ == '__main__':
    unittest.main()
