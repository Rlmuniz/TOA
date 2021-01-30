import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from toa.data import get_airplane_data
from toa.models.aero.ground_effect_comp import GroundEffectComp


class TestGroundEffectComp(unittest.TestCase):

    def setUp(self):
        self.airplane = get_airplane_data('b734')

    def test_value(self):
        p = om.Problem()
        n = 1

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='h', val=4, units='m')

        p.model.add_subsystem('ground_effect', subsys=GroundEffectComp(num_nodes=n, airplane=self.airplane))

        p.model.connect('h', 'ground_effect.h')
        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('ground_effect.CLag'), 6.02, tolerance=.05)
        assert_near_equal(p.get_val('ground_effect.dalpha_zero'), 0.42, tolerance=.05)
        assert_near_equal(p.get_val('ground_effect.phi'), 0.63, tolerance=.05)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()