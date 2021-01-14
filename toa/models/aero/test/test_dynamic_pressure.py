import unittest
import numpy as np
import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp
from dymos.utils.testing_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

from toa.models.aero.dynamic_pressure_comp import DynamicPressureComp


class TestDynamicPressureComp(unittest.TestCase):

    def test_value(self):
        n = 5
        p = om.Problem()
        tas = np.linspace(0, 100, n)

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='tas', val=tas, units='m/s')
        ivc.add_output(name='h', val=0.0, units='m')
        p.model.add_subsystem('atmos', subsys=USatm1976Comp(num_nodes=1))
        p.model.add_subsystem(name='dyn_press', subsys=DynamicPressureComp(num_nodes=n))

        p.model.connect('tas', 'dyn_press.tas')
        p.model.connect('h', 'atmos.h')
        p.model.connect('atmos.rho', 'dyn_press.rho')
        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('dyn_press.qbar'), 0.5 * p.get_val('atmos.rho', units='kg/m**3') * tas ** 2)

    def test_partials(self):
        n = 5
        p = om.Problem()
        tas = np.linspace(0, 100, n)

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='tas', val=tas, units='m/s')
        ivc.add_output(name='h', val=0.0, units='m')
        p.model.add_subsystem('atmos', subsys=USatm1976Comp(num_nodes=1))
        p.model.add_subsystem(name='dyn_press', subsys=DynamicPressureComp(num_nodes=n))

        p.model.connect('tas', 'dyn_press.tas')
        p.model.connect('h', 'atmos.h')
        p.model.connect('atmos.rho', 'dyn_press.rho')
        p.setup()
        p.run_model()

        cpd = p.check_partials(compact_print=False, out_stream=None)
        assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-4)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()