import numpy as np
import openmdao.api as om

from toa.data import Airplane
from toa.data import get_airplane_data


class RotationEOM(om.ExplicitComponent):
    """Models the rotation phase (2 DoF) in the takeoff run."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane']
        ar = np.arange(nn)
        zz = np.zeros(nn)
        ones = np.ones(nn)

        self.add_input(name='thrust', val=ones, desc='Engine total thrust',
                       units='N')
        self.add_input(name='lift', val=ones, desc='Lift', units='N')
        self.add_input(name='drag', val=ones, desc='Drag force', units='N')
        self.add_input(name='moment', val=ones, desc='Aerodynamic moment',
                       units='N*m')
        self.add_input(name='V', val=ones, desc='Body x axis velocity',
                       units='m/s')
        self.add_input(name='mass', val=ones, desc='Airplane mass', units='kg')
        self.add_input(name='alpha', val=ones, desc='Angle of attack',
                       units='rad')
        self.add_input(name='q', val=ones, desc='Pitch rate', units='rad/s')
        self.add_input(name='rw_slope', val=0.0, desc='Runway slope', units='rad')
        self.add_input(name='grav', val=0.0, desc='Gravity acceleration',
                       units='m/s**2')

        self.add_output(name='x_dot', val=ones, desc='Derivative of horizontal position',
                        units='m/s')
        self.add_output(name='h_dot', val=ones, desc='Derivative of vertical position',
                        units='m/s')
        self.add_output(name='v_dot', val=ones, desc="Body x axis acceleration",
                        units='m/s**2')
        self.add_output(name='theta_dot', val=ones, desc="Alpha derivative",
                        units='rad/s')
        self.add_output(name='q_dot', val=ones, desc="Pitch rate derivative",
                        units='rad/s**2')
        self.add_output(name='f_mg', val=ones, desc='Main wheel reaction force',
                        units='N')

        self.declare_partials(of='v_dot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='v_dot', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='rw_slope', rows=ar, cols=zz)

        self.declare_partials(of='x_dot', wrt='V', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='x_dot', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='x_dot', wrt='q', rows=ar, cols=ar)

        self.declare_partials(of='h_dot', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='h_dot', wrt='q', rows=ar, cols=ar)

        self.declare_partials(of='q_dot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='q_dot', wrt='moment', rows=ar, cols=ar, val=1 / airplane.inertia.iy)
        self.declare_partials(of='q_dot', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='q_dot', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='q_dot', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='q_dot', wrt='rw_slope', rows=ar, cols=zz)

        self.declare_partials(of='theta_dot', wrt='q', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='f_mg', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='f_mg', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='f_mg', wrt='lift', rows=ar, cols=ar, val=-1.0)
        self.declare_partials(of='f_mg', wrt='rw_slope', rows=ar, cols=zz)


    def compute(self, inputs, outputs, **kwargs):
        thrust = inputs['thrust']
        lift = inputs['lift']
        drag = inputs['drag']
        moment = inputs['moment']
        V = inputs['V']
        mass = inputs['mass']
        alpha = inputs['alpha']
        q = inputs['q']
        grav = inputs['grav']
        rw_slope = inputs['rw_slope']
        airplane = self.options['airplane']

        mu = 0.025
        xmg = airplane.landing_gear.main.x
        zm = airplane.landing_gear.main.z
        zt = airplane.engine.zt
        weight = mass * grav
        cosslope = np.cos(rw_slope)
        sinslope = np.sin(rw_slope)
        cosalpha = np.cos(alpha)
        sinalpha = np.sin(alpha)

        f_mg = weight * cosslope - lift
        f_rr = mu * f_mg
        m_mg = - xmg * f_mg - f_rr * zm

        outputs['v_dot'] = (thrust * cosalpha - drag - f_rr - weight * sinslope) / mass
        outputs['x_dot'] = V - q * xmg * sinalpha
        outputs['h_dot'] = q * xmg * cosalpha
        outputs['q_dot'] = (moment + m_mg + thrust * zt) / airplane.inertia.iy
        outputs['theta_dot'] = q
        outputs['f_mg'] = f_mg

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane']
        alpha = inputs['alpha']
        mass = inputs['mass']
        thrust = inputs['thrust']
        lift = inputs['lift']
        drag = inputs['drag']
        rw_slope = inputs['rw_slope']
        grav = inputs['grav']
        q = inputs['q']

        mu = 0.025
        xmg = airplane.landing_gear.main.x
        zm = airplane.landing_gear.main.z
        zt = airplane.engine.zt

        cosalpha = np.cos(alpha)
        sinalpha = np.sin(alpha)
        cosslope = np.cos(rw_slope)
        sinslope = np.sin(rw_slope)

        partials['v_dot', 'thrust'] = cosalpha / mass
        partials['v_dot', 'alpha'] = -thrust * sinalpha / mass
        partials['v_dot', 'drag'] = -1 / mass
        partials['v_dot', 'mass'] = (drag - lift * mu - thrust * cosalpha) / mass ** 2
        partials['v_dot', 'grav'] = -mu*cosslope - sinslope
        partials['v_dot', 'lift'] = mu / mass
        partials['v_dot', 'rw_slope'] = grav*(mu*sinslope - cosslope)

        partials['x_dot', 'q'] = -xmg * sinalpha
        partials['x_dot', 'alpha'] = -q * xmg * cosalpha

        partials['h_dot', 'q'] = xmg * cosalpha
        partials['h_dot', 'alpha'] = -q * xmg * sinalpha

        partials['q_dot', 'thrust'] = zt / airplane.inertia.iy
        partials['q_dot', 'mass'] = -grav*(mu*zm + xmg)*cosslope / airplane.inertia.iy
        partials['q_dot', 'grav'] = -mass*(mu*zm + xmg)*cosslope / airplane.inertia.iy
        partials['q_dot', 'lift'] = (mu*zm + xmg) / airplane.inertia.iy
        partials['q_dot', 'rw_slope'] = grav*mass*(mu*zm + xmg)*sinslope / airplane.inertia.iy

        partials['f_mg', 'mass'] = grav * cosslope
        partials['f_mg', 'grav'] = mass * cosslope
        partials['f_mg', 'rw_slope'] = -grav * mass * sinslope


if __name__ == '__main__':
    prob = om.Problem()
    airplane = get_airplane_data('b734')
    num_nodes = 1
    prob.model.add_subsystem('comp', RotationEOM(num_nodes=1, airplane=airplane))

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    prob.check_partials(compact_print=True, show_only_incorrect=True)
