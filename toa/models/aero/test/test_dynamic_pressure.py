import unittest
import numpy as np
import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp
from openmdao.utils.assert_utils import assert_near_equal

from toa.models.aero.dynamic_pressure_comp import DynamicPressureComp


class TestDynamicPressureComp(unittest.TestCase):

    def test_value(self):
        n = 5
        p = om.Problem()
        rho = 1.225
        p.model.add_subsystem(name='dyn_press', subsys=DynamicPressureComp(num_nodes=n))

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='tas', val=np.linspace(0, 100, n), units='m/s')
        ivc.add_output(name='h', val=0.0, units='m')

        p.model.add_subsystem('atmos', subsys=USatm1976Comp(num_nodes=1))

        p.model.connect('tas', 'dyn_press.tas')
        p.model.connect('h', 'atmos.h')
        p.model.connect('atmos.rho', 'dyn_press.rho')

        p.setup()
        p.run_model()
        print(f"Outer function rho: {p.get_val('dyn_press.rho')}")
        print(p.get_val('dyn_press.tas'))
        print(p.get_val('atmos.rho'))
        assert_near_equal(p.get_val('dyn_press.qbar'), 0.5 * rho * ivc.get_val('tas') ** 2)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()