import numpy as np
import openmdao.api as om


class GroundRollEOM(om.ExplicitComponent):
    """Computes the ground run (1 DoF) with all wheels on the runway to the point of nose wheels lift-off."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.mu = 0.002
        # Inputs
        self.add_input(name='thrust', val=np.zeros(nn), desc='Engine total thrust', units='N')
        self.add_input(name='drag', val=np.zeros(nn), desc='Drag force', units='N')
        self.add_input(name='mlg_reaction', val=np.zeros(nn), desc='Main landing gear reaction', units='N')
        self.add_input(name='nlg_reaction', val=np.zeros(nn), desc='Nose landing gear reaction', units='N')
        self.add_input(name='rw_slope', val=0.0, desc='Runway slope', units='rad')
        self.add_input(name='grav', val=0.0, desc='Gravity acceleration', units='m/s**2')
        self.add_input(name='mass', val=np.zeros(nn), desc='Airplane mass', units='kg')
        self.add_input(name='alpha', val=np.zeros(nn), desc='Angle of attack', units='rad')

        # Outputs
        self.add_output(name='dXdt:v', val=np.zeros(nn), desc="Body x axis acceleration", units='m/s**2')

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='dXdt:v', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='mlg_reaction', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='nlg_reaction', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='rw_slope', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='dXdt:v', wrt='grav', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='dXdt:v', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='alpha', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        thrust = inputs['thrust']
        drag = inputs['drag']
        mlg_reaction = inputs['mlg_reaction']
        nlg_reaction = inputs['nlg_reaction']
        mass = inputs['mass']
        weight = inputs['grav'] * mass
        mu = self.mu

        sslope = np.sin(inputs['rw_slope'])
        calpha = np.cos(inputs['rw_slope'])

        outputs['dXdt:v'] = (thrust * calpha - drag - mu * (mlg_reaction + nlg_reaction) - weight * sslope) / mass

    def compute_partials(self, inputs, partials, **kwargs):
        thrust = inputs['thrust']
        drag = inputs['drag']
        mlg_reaction = inputs['mlg_reaction']
        nlg_reaction = inputs['nlg_reaction']
        mass = inputs['mass']
        grav = inputs['grav']
        mu = self.mu

        sslope = np.sin(inputs['rw_slope'])
        cslope = np.cos(inputs['rw_slope'])
        calpha = np.cos(inputs['alpha'])
        salpha = np.sin(inputs['alpha'])

        partials['dXdt:v', 'thrust'] = calpha / mass
        partials['dXdt:v', 'drag'] = -1 / mass
        partials['dXdt:v', 'mlg_reaction'] = - mu / mass
        partials['dXdt:v', 'nlg_reaction'] = - mu / mass
        partials['dXdt:v', 'rw_slope'] = - grav * cslope
        partials['dXdt:v', 'grav'] = - sslope
        partials['dXdt:v', 'mass'] = - grav * sslope / mass - (
                -drag - mu * (mlg_reaction + nlg_reaction) - grav * mass * sslope + thrust * calpha) / mass ** 2
        partials['dXdt:v', 'alpha'] = - thrust * salpha / mass