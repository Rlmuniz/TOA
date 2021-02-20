import openmdao.api as om
import numpy as np

from toa.data import Airplane
from toa.models.aero.drag_coef_comp import DragCoeffComp
from toa.models.aero.dynamic_pressure_comp import DynamicPressureComp
from toa.models.aero.flap_slat_comp import FlapSlatComp
from toa.models.aero.ground_effect_comp import GroundEffectComp
from toa.models.aero.lift_coef_comp import LiftCoeffAllWheelsOnGroundComp
from toa.models.aero.lift_coef_comp import LiftCoeffComp
from toa.models.aero.lift_coef_ground_correction_comp import LiftCoeffGroundCorrectionComp
from toa.models.aero.lift_drag_moment_comp import LiftDragMomentComp
from toa.models.aero.moment_coef_comp import MomentCoeffAllWheelsOnGroundComp
from toa.models.aero.moment_coef_comp import MomentCoeffComp


class AerodynamicsGroup(om.Group):
    """Computes the lift and drag forces on the aircraft."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')
        self.options.declare('landing_gear', default=True,
                             desc='Accounts landing gear drag')
        self.options.declare('AllWheelsOnGround', default=True)

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane']
        landing_gear = self.options['landing_gear']
        all_wheels_on_ground = self.options['AllWheelsOnGround']

        self.add_subsystem(name='flap_slat',
                           subsys=FlapSlatComp(airplane=airplane),
                           promotes_inputs=['flap_angle'], promotes_outputs=['CLmax'])

        self.add_subsystem(name='ground_effect', subsys=GroundEffectComp(num_nodes=nn, airplane=airplane))

        if all_wheels_on_ground:
            self.add_subsystem(name='cl_comp',
                               subsys=LiftCoeffAllWheelsOnGroundComp(num_nodes=nn,
                                                                     airplane=airplane),
                               promotes_inputs=['alpha', 'de', 'dih'],
                               promotes_outputs=['CL'])
            self.add_subsystem(name='cm_comp',
                               subsys=MomentCoeffAllWheelsOnGroundComp(num_nodes=nn,
                                                                       airplane=airplane),
                               promotes_inputs=['alpha', 'de', 'dih'],
                               promotes_outputs=['Cm'])
        else:
            self.add_subsystem(name='cl_comp',
                               subsys=LiftCoeffComp(num_nodes=nn, airplane=airplane),
                               promotes_inputs=['alpha', 'de', 'tas', 'q', 'dih'],
                               promotes_outputs=['CL'])
            self.add_subsystem(name='cm_comp',
                               subsys=MomentCoeffComp(num_nodes=nn, airplane=airplane),
                               promotes_inputs=['alpha', 'de', 'tas', 'q', 'dih'],
                               promotes_outputs=['Cm'])

        self.connect('flap_slat.CL0', 'cl_comp.CL0')
        self.connect('flap_slat.CLa', 'cl_comp.CLa')

        self.add_subsystem(name='cl_ground_corr', subsys=LiftCoeffGroundCorrectionComp(num_nodes=nn),
                           promotes_inputs=['CL'])

        self.connect('ground_effect.CLag', 'cl_ground_corr.CLag')
        self.connect('ground_effect.dalpha_zero', 'cl_ground_corr.dalpha_zero')
        self.connect('flap_slat.CLa', 'cl_ground_corr.CLa')

        self.add_subsystem(name='cd_comp',
                           subsys=DragCoeffComp(num_nodes=nn, airplane=airplane,
                                                landing_gear=landing_gear, partial_coloring=True),
                           promotes_inputs=['flap_angle', 'grav', 'mass'],
                           promotes_outputs=['CD'])

        self.connect('cl_ground_corr.CLg', 'cd_comp.CL')
        self.connect('ground_effect.phi', 'cd_comp.phi')

        self.add_subsystem(name='dyn_press',
                           subsys=DynamicPressureComp(num_nodes=nn),
                           promotes_inputs=['rho', 'tas'],
                           promotes_outputs=['qbar'])

        self.add_subsystem(name='lift_drag_moment_comp',
                           subsys=LiftDragMomentComp(num_nodes=nn,
                                                     airplane=airplane),
                           promotes_inputs=['CD', 'Cm', 'qbar'],
                           promotes_outputs=['L', 'D', 'M'])

        self.connect('cl_ground_corr.CLg', 'lift_drag_moment_comp.CL')
        self.set_input_defaults('flap_angle', val=0.0, units='deg')
