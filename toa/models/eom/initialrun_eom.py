import numpy as np
import openmdao.api as om

from toa.data import Airplane
from toa.data import get_airplane_data


class InitialRunEOM(om.ExplicitComponent):
    """Computes the ground run (1 DoF) with all wheels on the runway to the point of nose wheels lift-off."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)
        ones = np.ones(nn)

        # Inputs
        self.add_input(name='thrust', val=ones, desc='Engine total thrust',
                       units='N')
        self.add_input(name='lift', val=ones, desc='Lift', units='N')
        self.add_input(name='drag', val=ones, desc='Drag force', units='N')
        self.add_input(name='moment', val=ones, desc='Aerodynamic moment',
                       units='N*m')
        self.add_input(name='V', val=ones, desc='Body x axis velocity',
                       units='m/s')
        self.add_input(name='mass', val=ones, desc='Airplane mass', units='kg')
        self.add_input(name='rw_slope', val=0.0, desc='Runway slope', units='rad')
        self.add_input(name='grav', val=0.0, desc='Gravity acceleration',
                       units='m/s**2')
        self.add_input(name='Vw', val=0.0,
                       desc='Wind speed along the runway, defined as positive for a headwind',
                       units='m/s')
        self.add_input(name='alpha', val=ones, desc='Angle of attack', units='rad')

        # Outputs
        self.add_output(name='v_dot', val=ones, desc="Body x axis acceleration",
                        units='m/s**2')
        self.add_output(name='x_dot', val=ones, desc="Derivative of position",
                        units='m/s')
        self.add_output(name='f_ng', val=ones, desc="Nose wheel reaction force",
                        units='N')
        self.add_output(name='f_mg', val=ones, desc="Main wheel reaction force",
                        units='N')

        # Partials
        self.declare_partials(of='v_dot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='v_dot', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='rw_slope', rows=ar, cols=zz)

        self.declare_partials(of='x_dot', wrt='V', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='x_dot', wrt='Vw', rows=ar, cols=zz, val=-1.0)

        self.declare_partials(of='f_ng', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='f_ng', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='f_ng', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='f_ng', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='f_ng', wrt='rw_slope', rows=ar, cols=zz)

        self.declare_partials(of='f_mg', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='f_mg', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='f_mg', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='f_mg', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='f_mg', wrt='rw_slope', rows=ar, cols=zz)

    def compute(self, inputs, outputs, **kwargs):
        thrust = inputs['thrust']
        lift = inputs['lift']
        drag = inputs['drag']
        moment = inputs['moment']
        V = inputs['V']
        mass = inputs['mass']
        grav = inputs['grav']
        rw_slope = inputs['rw_slope']
        alpha = inputs['alpha']
        Vw = inputs['Vw']
        airplane = self.options['airplane']

        mu_mg = 0.025
        mu_ng = mu_mg

        xmg = airplane.landing_gear.main.x
        xng = airplane.landing_gear.nose.x
        weight = mass * grav
        cosslope = np.cos(rw_slope)
        sinslope = np.sin(rw_slope)
        cosalpha = np.cos(alpha)

        f_ng = (-moment + xmg * (cosslope * weight - lift)) / (xmg + xng)
        f_mg = (moment + xng * (cosslope * weight - lift)) / (xmg + xng)

        f_rr = mu_mg * f_mg + mu_ng * f_ng

        outputs['v_dot'] = (thrust * cosalpha - drag - f_rr - weight * sinslope) / mass
        outputs['x_dot'] = V - Vw
        outputs['f_ng'] = f_ng
        outputs['f_mg'] = f_mg

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane']
        alpha = inputs['alpha']
        mass = inputs['mass']
        thrust = inputs['thrust']
        drag = inputs['drag']
        moment = inputs['moment']
        lift = inputs['lift']
        rw_slope = inputs['rw_slope']
        grav = inputs['grav']

        xmg = airplane.landing_gear.main.x
        xng = airplane.landing_gear.nose.x
        mu_mg = 0.025
        mu_ng = mu_mg

        cosalpha = np.cos(alpha)
        sinalpha = np.sin(alpha)
        cosslope = np.cos(rw_slope)
        sinslope = np.sin(rw_slope)

        partials['v_dot', 'thrust'] = cosalpha / mass
        partials['v_dot', 'alpha'] = -thrust * sinalpha / mass
        partials['v_dot', 'drag'] = -1 / mass
        partials['v_dot', 'mass'] = (drag * xmg + drag * xng - lift * mu_mg * xng - lift * mu_ng * xmg
                                     + moment * mu_mg - moment * mu_ng - thrust * xmg * cosalpha
                                     - thrust * xng * cosalpha) / (mass ** 2 * (xmg + xng))
        partials['v_dot', 'grav'] = -(mu_mg * xng * cosslope
                                      + mu_ng * xmg * cosslope
                                      + (xmg + xng) * sinslope) / (xmg + xng)
        partials['v_dot', 'lift'] = (mu_mg * xng + mu_ng * xmg) / (mass * (xmg + xng))
        partials['v_dot', 'moment'] = (-mu_mg + mu_ng) / (mass * (xmg + xng))
        partials['v_dot', 'rw_slope'] = grav * (mu_mg * xng * sinslope + mu_ng * xmg * sinslope
                                                - (xmg + xng) * cosslope) / (xmg + xng)

        partials['f_ng', 'lift'] = -xmg / (xmg + xng)
        partials['f_ng', 'moment'] = -1 / (xmg + xng)
        partials['f_ng', 'mass'] = grav * xmg * cosslope / (xmg + xng)
        partials['f_ng', 'grav'] = mass * xmg * cosslope / (xmg + xng)
        partials['f_ng', 'rw_slope'] = -grav * mass * xmg * sinslope / (xmg + xng)

        partials['f_mg', 'lift'] = -xng / (xmg + xng)
        partials['f_mg', 'moment'] = 1 / (xmg + xng)
        partials['f_mg', 'mass'] = grav * xng * cosslope / (xmg + xng)
        partials['f_mg', 'grav'] = mass * xng * cosslope / (xmg + xng)
        partials['f_mg', 'rw_slope'] = -grav * mass * xng * sinslope / (xmg + xng)


if __name__ == '__main__':
    prob = om.Problem()
    airplane = get_airplane_data('b734')
    num_nodes = 1
    prob.model.add_subsystem('comp', InitialRunEOM(num_nodes=1, airplane=airplane))

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    prob.check_partials(compact_print=True, show_only_incorrect=True)