import openmdao.api as om


from toa.data.airplanes.airplanes import Airplanes
from toa.models.aero.coeff.drag_coef_comp import DragCoeffComp
from toa.models.aero.coeff.lift_coef_comp import LiftCoeffAllWheelsOnGroundComp
from toa.models.aero.coeff.lift_coef_comp import LiftCoeffComp
from toa.models.aero.coeff.moment_coef_comp import MomentCoeffAllWheelsOnGroundComp
from toa.models.aero.coeff.moment_coef_comp import MomentCoeffComp


class AerodynamicsCoefficientsGroup(om.Group):
    """Computes the lift and drag forces on the aircraft."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=Airplanes, desc='Class containing all airplane data')
        self.options.declare('landing_gear', default=True, desc='Accounts landing gear drag')
        self.options.declare('AllWheelsOnGround', default=True)

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']

        if self.options['AllWheelsOnGround']:
            self.add_subsystem(name='lift_coef_comp',
                               subsys=LiftCoeffAllWheelsOnGroundComp(num_nodes=nn, airplane_data=airplane),
                               promotes_inputs=['alpha', 'de'],
                               promotes_outputs=['CL'])
            self.add_subsystem(name='moment_coef_comp',
                               subsys=MomentCoeffAllWheelsOnGroundComp(num_nodes=nn, airplane_data=airplane),
                               promotes_inputs=['alpha', 'de'],
                               promotes_outputs=['Cm'])
        else:
            self.add_subsystem(name='lift_coef_comp',
                               subsys=LiftCoeffComp(num_nodes=nn, airplane_data=airplane),
                               promotes_inputs=['alpha', 'de', 'q', 'tas'],
                               promotes_outputs=['CL'])
            self.add_subsystem(name='moment_coef_comp',
                               subsys=MomentCoeffComp(num_nodes=nn, airplane_data=airplane),
                               promotes_inputs=['alpha', 'de', 'q', 'tas'],
                               promotes_outputs=['Cm'])

        self.add_subsystem(name='drag_coef_comp',
                           subsys=DragCoeffComp(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['flap_angle', 'CL', 'mass', 'grav'],
                           promotes_outputs=['CD'])