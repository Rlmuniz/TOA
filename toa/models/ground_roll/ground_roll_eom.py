import numpy as np
import openmdao.api as om


class GroundRollEOM(om.ExplicitComponent):
    """Computes the ground run (1 DoF) with all wheels on the runway to the point of nose wheels lift-off."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='thrust', val=np.zeros(nn), desc='Engine total thrust', units='N')
        self.add_input(name='drag', val=np.zeros(nn), desc='Drag force', units='N')
        self.add_input(name='mlg_reaction', val=np.zeros(nn), desc='Main landing gear reaction', units='N')
        self.add_input(name='nlg_reaction', val=np.zeros(nn), desc='Nose landing gear reaction', units='N')
        self.add_input(name='mu', val=np.zeros(nn), desc='Friction coefficient', units=None)
        self.add_input(name='rw_slope', val=np.zeros(nn), desc='Runway slope', units='N')
        self.add_input(name='grav', val=np.zeros(nn), desc='Gravity acceleration', units='m/s**2')
        self.add_input(name='mass', val=np.zeros(nn), desc='Airplane mass', units='kg')
        self.add_input(name='alpha', val=np.zeros(nn), desc='Angle of attack', units='rad')

        # Outputs
        self.add_output(name='v_dot', val=np.zeros(nn), desc="Body x axis acceleration", units='m/s**2')

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='v_dot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='mlg_reaction', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='nlg_reaction', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='rw_slope', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='grav', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='alpha', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        thrust = inputs['thrust']
        drag = inputs['drag']
        mlg_reaction = inputs['mlg_reaction']
        nlg_reaction = inputs['nlg_reaction']
        mass = inputs['mass']
        weight = inputs['grav'] * mass
        mu = inputs['mu']

        sslope = np.sin(inputs['rw_slope'])
        calpha = np.cos(inputs['rw_slope'])

        outputs['v_dot'] = (thrust * calpha - drag - mu * (mlg_reaction + nlg_reaction) - weight * sslope) / mass

    def compute_partials(self, inputs, partials, **kwargs):
        thrust = inputs['thrust']
        drag = inputs['drag']
        mlg_reaction = inputs['mlg_reaction']
        nlg_reaction = inputs['nlg_reaction']
        mass = inputs['mass']
        grav = inputs['grav']
        mu = inputs['mu']

        sslope = np.sin(inputs['rw_slope'])
        cslope = np.cos(inputs['rw_slope'])
        calpha = np.cos(inputs['alpha'])
        salpha = np.sin(inputs['alpha'])

        partials['v_dot', 'thrust'] = calpha / mass
        partials['v_dot', 'drag'] = -1 / mass
        partials['v_dot', 'mlg_reaction'] = - mu / mass
        partials['v_dot', 'nlg_reaction'] = - mu / mass
        partials['v_dot', 'rw_slope'] = - grav * cslope
        partials['v_dot', 'grav'] = - sslope
        partials['v_dot', 'mass'] = - grav * sslope / mass - (
                -drag - mu * (mlg_reaction + nlg_reaction) - grav * mass * sslope + thrust * calpha) / mass ** 2
        partials['v_dot', 'alpha'] = - thrust * salpha / mass