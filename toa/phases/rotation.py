import dymos as dm

from toa.ode.rotation import RotationODE
from scipy.constants import degree


class RotationPhase(dm.Phase):

    def initialize(self):
        super(RotationPhase, self).initialize()

        self.options['ode_class'] = RotationODE

        self.set_time_options(units='s')

        self.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='rotation_eom.v_dot',
                       targets=['v'], lower=0)
        self.add_state('x', fix_initial=False, fix_final=False, units='m', rate_source='v', upper=3000)
        self.add_state('pitch_rate', fix_initial=True, fix_final=False, units='rad/s', rate_source='rotation_eom.q_dot',
                       targets=['pitch_rate'])
        self.add_state('alpha', fix_initial=True, fix_final=False, units='rad', rate_source='pitch_rate',
                       targets=['alpha'], lower=0, upper=20 * degree)

        self.add_control(name='de', units='rad', lower=-0.5, upper=0.5, targets=['de'], opt=True)

        self.add_path_constraint('mlg_reaction', lower=0, units='N')
        self.add_boundary_constraint('mlg_reaction', loc='final', constraint_name='liftoff', lower=0.1, units='N')
