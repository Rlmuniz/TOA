import numpy as np
import openmdao.api as om

from toa.data import Airplane


class InitialRunEOM(om.ExplicitComponent):
    """Computes the ground run (1 DoF) with all wheels on the runway to the point of nose wheels lift-off."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        # Inputs
        self.add_input(name='thrust', val=np.zeros(nn), desc='Engine total thrust',
                       units='N')
        self.add_input(name='lift', val=np.zeros(nn), desc='Lift', units='N')
        self.add_input(name='drag', val=np.zeros(nn), desc='Drag force', units='N')
        self.add_input(name='moment', val=np.zeros(nn), desc='Aerodynamic moment',
                       units='N*m')
        self.add_input(name='V', val=np.zeros(nn), desc='Body x axis velocity',
                       units='m/s')
        self.add_input(name='mass', val=np.zeros(nn), desc='Airplane mass', units='kg')
        self.add_input(name='rw_slope', val=0.0, desc='Runway slope', units='rad')
        self.add_input(name='grav', val=0.0, desc='Gravity acceleration',
                       units='m/s**2')
        self.add_input(name='alpha', val=0.0, desc='Angle of attack', units='rad')

        # Outputs
        self.add_output(name='v_dot', val=np.zeros(nn), desc="Body x axis acceleration",
                        units='m/s**2')
        self.add_output(name='x_dot', val=np.zeros(nn), desc="Derivative of position",
                        units='m/s')
        self.add_output(name='f_ng', val=np.zeros(nn), desc="Nose wheel reaction force",
                        units='N')
        self.add_output(name='f_mg', val=np.zeros(nn), desc="Main wheel reaction force",
                        units='N')

        # Partials
        self.declare_partials(of='v_dot', wrt=['*'], method='fd')
        self.declare_partials(of='x_dot', wrt='V', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='f_ng', wrt=['*'], method='fd')
        self.declare_partials(of='f_mg', wrt=['*'], method='fd')

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
        airplane = self.options['airplane']

        mu_mg = 0.002
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
        outputs['x_dot'] = V
        outputs['f_ng'] = f_ng
        outputs['f_mg'] = f_mg