import dymos as dm

from toa.ode.initialrun_ode import GroundRollODE
from scipy.constants import degree


class GroundRollPhase(dm.Phase):

    def initialize(self):
        super(GroundRollPhase, self).initialize()

        self.options['ode_class'] = GroundRollODE

        self.set_time_options(fix_initial=True, units='s')

        self.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='ground_run_eom.v_dot',
                       targets=['v'], lower=0)
        self.add_state('x', fix_initial=True, fix_final=False, units='m', rate_source='v', lower=0, upper=3000)

        self.add_control(name='de', units='rad', lower=-30 * degree, upper=30 * degree, targets=['de'], opt=True)

        self.add_path_constraint('mlg_reaction', lower=0, units='N')
        self.add_path_constraint('nlg_reaction', lower=0, units='N')
        self.add_boundary_constraint('nlg_reaction', loc='final', constraint_name='nlg_liftoff', lower=0.1, units='N')

        self.add_objective('mass', loc='initial', scaler=-1)
