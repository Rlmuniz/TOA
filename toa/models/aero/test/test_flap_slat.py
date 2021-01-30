import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from toa.data import get_airplane_data
from toa.models.aero.flap_slat_comp import FlapSlatComp


class TestFlapSlatComp(unittest.TestCase):

    def setUp(self):
        self.airplane = get_airplane_data('b734')

    def test_flap_0(self):
        p = om.Problem()

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='flap_angle', val=0.0, units='deg')

        p.model.add_subsystem('flap_slat', subsys=FlapSlatComp(airplane=self.airplane))

        p.model.connect('flap_angle', 'flap_slat.flap_angle')
        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('flap_slat.CL0'), 0.12, tolerance=.01)
        assert_near_equal(p.get_val('flap_slat.CLmax'), 1.42, tolerance=.01)
        assert_near_equal(p.get_val('flap_slat.CLa'), 5.47, tolerance=.01)
        assert_near_equal(p.get_val('flap_slat.alpha_max'), 13.58, tolerance=.01)

    def test_flap_25(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(),
                                    promotes_outputs=['*'])
        ivc.add_output(name='flap_angle', val=25, units='deg')

        p.model.add_subsystem('flap_slat', subsys=FlapSlatComp(airplane=self.airplane))

        p.model.connect('flap_angle', 'flap_slat.flap_angle')
        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('flap_slat.CL0'), 0.90, tolerance=.01)
        assert_near_equal(p.get_val('flap_slat.CLmax'), 2.44, tolerance=.01)
        assert_near_equal(p.get_val('flap_slat.CLa'), 5.76, tolerance=.01)
        assert_near_equal(p.get_val('flap_slat.alpha_max'), 15.28, tolerance=.01)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()