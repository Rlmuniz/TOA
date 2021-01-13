import unittest
import numpy as np
import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp
from openap import Thrust
from openmdao.utils.assert_utils import assert_near_equal

from toa.data.airplanes.examples import b747


class TestPropulsionGround(unittest.TestCase):

    def test_thrust_value(self):
        n = 5
        p = om.Problem()
        tas = np.linspace(0, 100, n)
        alt = 0

        p.model.add_subsystem(name='propulsion', subsys=PropulsionGroup(num_nodes=n, airplane_data=b747))
        p.model.add_subsystem(name='atmos', subsys=USatm1976Comp(num_nodes=1))
        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='tas', val=tas, units='m/s')
        ivc.add_output(name='alt', val=alt, units='m')

        p.model.connect('tas', 'propulsion.tas')
        p.model.connect('alt', ['atmos.h', 'propulsion.fuel_flow.elevation'])
        p.model.connect('atmos.pres', 'propulsion.thrust_comp.p_amb')
        p.model.connect('atmos.sos', 'propulsion.sos')

        p.setup()
        p.run_model()

        calculated_thrust = p.get_val('propulsion.thrust')
        print(f"Mach: {p.get_val('propulsion.mach_comp.mach')}")
        print(f"Press Amb: {p.get_val('propulsion.thrust_comp.p_amb')}")
        print(f"Tas: {p.get_val('propulsion.tas')}")
        print(f"sos: {p.get_val('propulsion.sos')}")
        print(f"thrust: {calculated_thrust}")

        thrust = Thrust(ac='B744', eng='PW4062')
        reference_thrust = [thrust.takeoff(tas=v, alt=alt) for v in tas]
        assert_near_equal(calculated_thrust, reference_thrust)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()