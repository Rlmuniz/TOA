import unittest
import numpy as np
import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp
from openap import Drag
from openmdao.utils.assert_utils import assert_near_equal

from toa.data.airplanes.examples import b747
from toa.models.aero.aerodynamics import AerodynamicsGroup


class TestAerodynamicGroud(unittest.TestCase):

    def test_drag(self):
        n = 10
        p = om.Problem()
        tas_kt = np.linspace(0, 100, n) * 1.94384
        mass = 200000
        flap_angle = 10
        path_angle = 0
        alt = 0

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='tas', val=tas_kt, units='kn')
        ivc.add_output(name='mass', val=mass, units='kg')
        ivc.add_output(name='flap_angle', val=flap_angle, units='deg')
        ivc.add_output(name='alt', val=alt, units='m')
        ivc.add_output(name='grav', val=9.80665, units='m/s**2')

        p.model.add_subsystem('atmos', subsys=USatm1976Comp(num_nodes=1))

        p.model.add_subsystem('cl_comp', subsys=om.ExecComp('CL=(mass * grav * np.cos(flight_path))/(qbar * S)',
                                                            mass={'value': 0.0, 'units': 'kg'},
                                                            grav={'value': 0.0, 'units': 'm/s**2'},
                                                            flight_path={'value': 0.0, 'units': 'rad'},
                                                            qbar={'value':np.repeat(0.0, n), 'units': 'Pa'},
                                                            S={'value': 0.0, 'units':'m**2'},
                                                            cl={'value': np.repeat(0.0, n), 'units':'None'}))
        p.model.add_subsystem('aero', subsys=AerodynamicsGroup(num_nodes=n, airplane_data=b747))

        p.model.connect('tas', 'aero.tas')
        p.model.connect('alt', 'atmos.h')
        p.model.connect('atmos.rho', 'aero.rho')
        p.model.connect('flap_angle', 'aero.flap_angle')
        p.model.connect('aero.qbar', 'cl_comp.qbar')
        p.model.connect('mass', ['aero.mass', 'cl_comp.mass'])
        p.model.connect('grav', ['aero.grav', 'cl_comp.grav'])
        p.model.connect('aero.coeff_comp.CL', 'cl_comp.CL')
        p.model.connect('cl_comp.CL', 'aero.coeff_comp.CL')
        p.setup()

        p.set_val('cl_comp.S', b747.wing.area)
        p.set_val('cl_comp.flight_path', path_angle)
        p.run_model()

        print(p.get_val('aero.D'))
        drag = Drag(ac='B744')
        reference_drag = [drag.nonclean(mass=mass, tas=v, alt=alt, flap_angle=flap_angle,
                  path_angle=path_angle, landing_gear=True) for v in tas_kt]
        print(reference_drag)
        assert_near_equal(p.get_val('aero.D'), reference_drag)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
