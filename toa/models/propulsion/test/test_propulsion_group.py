import unittest
import numpy as np
import openmdao.api as om
from openap import FuelFlow
from openap import Thrust
from dymos.models.atmosphere import USatm1976Comp
from openmdao.utils.assert_utils import assert_near_equal

from toa.data import get_airplane_data
from toa.models.propulsion.propulsion_group import PropulsionGroup


class TestPropulsionGround(unittest.TestCase):

    def setUp(self) -> None:
        self.airplane_744 = get_airplane_data('b744')
        self.airplane_734 = get_airplane_data('b734')

    def test_thrust_value_b744(self):
        nn = 5
        p = om.Problem()
        tas = np.linspace(0, 100, nn)
        tas_kt = tas*1.94384
        alt = 0

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='tas', val=tas, units='m/s')
        ivc.add_output(name='alt', val=alt, units='m')
        p.model.connect('alt', ['atmos.h', 'propulsion.elevation'])
        p.model.connect('tas', ['propulsion.tas'])

        p.model.add_subsystem(name='atmos', subsys=USatm1976Comp(num_nodes=1))
        p.model.connect('atmos.sos', 'propulsion.sos')
        p.model.connect('atmos.pres', ['propulsion.p_amb'])

        p.model.add_subsystem(name='propulsion', subsys=PropulsionGroup(num_nodes=nn, airplane=self.airplane_744))

        p.setup()
        p.run_model()

        thrust = Thrust(ac='B744', eng='PW4062')
        reference_thrust = [thrust.takeoff(tas=v, alt=alt) for v in tas_kt]
        assert_near_equal(p.get_val('propulsion.thrust'), reference_thrust, tolerance=0.01)

    def test_thrust_value_b734(self):
        nn = 5
        p = om.Problem()
        tas = np.linspace(0, 100, nn)
        tas_kt = tas*1.94384
        alt = 0

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='tas', val=tas, units='m/s')
        ivc.add_output(name='alt', val=alt, units='m')
        p.model.connect('alt', ['atmos.h', 'propulsion.elevation'])
        p.model.connect('tas', ['propulsion.tas'])

        p.model.add_subsystem(name='atmos', subsys=USatm1976Comp(num_nodes=1))
        p.model.connect('atmos.sos', 'propulsion.sos')
        p.model.connect('atmos.pres', ['propulsion.p_amb'])

        p.model.add_subsystem(name='propulsion', subsys=PropulsionGroup(num_nodes=nn, airplane=self.airplane_734))

        p.setup()
        p.run_model()

        thrust = Thrust(ac='B734')
        reference_thrust = [thrust.takeoff(tas=v, alt=alt) for v in tas_kt]
        assert_near_equal(p.get_val('propulsion.thrust'), reference_thrust, tolerance=0.001)

    def test_fuel_value_734(self):
        nn = 5
        p = om.Problem()
        tas = np.linspace(0, 100, nn)
        tas_kt = tas*1.94384
        alt = 0

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='tas', val=tas, units='m/s')
        ivc.add_output(name='alt', val=alt, units='m')
        p.model.connect('alt', ['atmos.h', 'propulsion.elevation'])
        p.model.connect('tas', ['propulsion.tas'])

        p.model.add_subsystem(name='atmos', subsys=USatm1976Comp(num_nodes=1))

        p.model.add_subsystem(name='propulsion', subsys=PropulsionGroup(num_nodes=nn, airplane=self.airplane_734))

        p.model.connect('atmos.sos', 'propulsion.sos')
        p.model.connect('atmos.pres', ['propulsion.p_amb'])
        p.setup()
        p.run_model()

        ff = FuelFlow(ac='B734')
        reference_ff = [-1*ff.takeoff(tas=v, alt=alt) for v in tas_kt]
        assert_near_equal(p.get_val('propulsion.m_dot'), reference_ff, tolerance=0.001)

    def test_fuel_value_744(self):
        nn = 5
        p = om.Problem()
        tas = np.linspace(0, 100, nn)
        tas_kt = tas*1.94384
        alt = 0

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='tas', val=tas, units='m/s')
        ivc.add_output(name='alt', val=alt, units='m')
        p.model.connect('alt', ['atmos.h', 'propulsion.elevation'])
        p.model.connect('tas', ['propulsion.tas'])

        p.model.add_subsystem(name='atmos', subsys=USatm1976Comp(num_nodes=1))

        p.model.add_subsystem(name='propulsion', subsys=PropulsionGroup(num_nodes=nn, airplane=self.airplane_744))

        p.model.connect('atmos.sos', 'propulsion.sos')
        p.model.connect('atmos.pres', ['propulsion.p_amb'])
        p.setup()
        p.run_model()

        ff = FuelFlow(ac='B744', eng='PW4062')
        reference_ff = [-1*ff.takeoff(tas=v, alt=alt) for v in tas_kt]
        assert_near_equal(p.get_val('propulsion.m_dot'), reference_ff, tolerance=1e-5)



if __name__ == '__main__':  # pragma: no cover
    unittest.main()