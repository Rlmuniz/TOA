import numpy as np
import openmdao.api as om

from toa.data.airplanes.airplanes import Airplanes


class TransitionOEM(om.ExplicitComponent):
    """Models the transition phase (3 DoF) from liftoff to screen height."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=Airplanes, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='T', val=np.zeros(nn), desc='Engine total thrust', units='N')
        self.add_input(name='L', val=np.zeros(nn), desc='Lift', units='N')
        self.add_input(name='D', val=np.zeros(nn), desc='Drag force', units='N')
        self.add_input(name='M', val=np.zeros(nn), desc='Aerodynamic moment', units='N*m')
        self.add_input(name='V', val=np.zeros(nn), desc='Body x axis velocity', units='m/s')
        self.add_input(name='mass', val=np.zeros(nn), desc='Airplane mass', units='kg')
        self.add_input(name='alpha', val=np.zeros(nn), desc='Angle of attack', units='rad')
        self.add_input('q', val=np.zeros(nn), desc='Pitch rate', units='rad/s')
        self.add_input(name='rw_slope', val=0.0, desc='Runway slope', units='rad')
        self.add_input(name='grav', val=0.0, desc='Gravity acceleration', units='m/s**2')

        self.add_output('dXdt:x', val=np.zeros(nn), desc='Derivative of position', units='m/s')
        self.add_output(name='dXdt:v', val=np.zeros(nn), desc="Body x axis acceleration", units='m/s**2')
        self.add_output(name='dXdt:alpha', val=np.zeros(nn), desc="Alpha derivative", units='rad/s')
        self.add_output(name='dXdt:q', val=np.zeros(nn), desc="Pitch rate derivative", units='rad/s**2')
        self.add_output(name='rf_mainwheel', val=np.zeros(nn), desc='Main wheel reaction force', units='N')

        self.declare_partials(of='dXdt:v', wrt=['*'], method='fd')
        self.declare_partials(of='dXdt:x', wrt='V', method='fd')
        self.declare_partials(of='dXdt:alpha', wrt='q', method='fd')
        self.declare_partials(of='dXdt:q', wrt=['*'], method='fd')
        self.declare_partials(of='rf_mainwheel', wrt=['*'], method='fd')

    def compute(self, inputs, outputs, **kwargs):
        thrust = inputs['thrust']
        lift = inputs['lift']
        drag = inputs['drag']
        moment = inputs['moment']
        V = inputs['V']
        mass = inputs['mass']
        alpha = inputs['alpha']
        q = inputs['q']
        grav, = inputs['grav']
        rw_slope, = inputs['rw_slope']
        airplane = self.options['airplane_data']

        mu = 0.002

        weight = mass * grav

        rf_mainwheel = weight * np.cos(rw_slope) - lift
        f_rr = mu * rf_mainwheel
        m_mainwheel = airplane.landing_gear.x_mg * rf_mainwheel

        outputs['dXdt:v'] = (thrust * np.cos(alpha) - drag - f_rr - weight * np.sin(rw_slope)) / mass
        outputs['dXdt:x'] = V
        outputs['dXdt:q'] = (moment + m_mainwheel) / airplane.inertia.Iy
        outputs['dXdt:alpha'] = q
        outputs['rf_mainwheel'] = rf_mainwheel