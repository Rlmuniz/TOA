import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt

from toa.data import get_airplane_data
from toa.ode.initialrun_ode import InitialRunODE
from toa.ode.rotation_ode import RotationODE
from dymos.examples.plotting import plot_results

from toa.runway import Runway

p = om.Problem(model=om.Group())

# set driver
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SLSQP'

runway = Runway(1800, 0.0, 0.0, 0.0, 0.0)
airplane = get_airplane_data('b744')

traj = p.model.add_subsystem('traj', dm.Trajectory())
initialrun = dm.Phase(ode_class=InitialRunODE,
                      transcription=dm.GaussLobatto(num_segments=20),
                      ode_init_kwargs={'airplane': airplane})
traj.add_phase(name='initialrun', phase=initialrun)

# Initial run states
initialrun.add_state(name='V', units='kn', rate_source='initial_run_eom.v_dot',
                     targets=['V'], fix_initial=True, fix_final=False, lower=0)
initialrun.add_state(name='x', units='m', rate_source='initial_run_eom.x_dot',
                     fix_initial=True, fix_final=False, lower=0)
initialrun.add_state(name='mass', units='kg', rate_source='prop.m_dot',
                     targets=['mass'], fix_initial=False, fix_final=False)

# Initial run parameters
initialrun.add_parameter(name='de', val=0.0, units='deg', desc='Elevator deflection',
                         lower=-30.0, upper=30.0, targets=['aero.de'], opt=True)
initialrun.add_parameter(name='elevation', val=0.0, units='m', desc='Runway elevation',
                         targets=['elevation'], opt=False, dynamic=False)
initialrun.add_parameter(name='rw_slope', val=0.0, units='rad', desc='Runway slope',
                         targets=['initial_run_eom.rw_slope'], opt=False, dynamic=False)

# Initial run path constraints
initialrun.add_path_constraint(name='initial_run_eom.f_mg',
                               constraint_name='f_mg_restriction', lower=0, units='N')
initialrun.add_path_constraint(name='initial_run_eom.f_ng',
                               constraint_name='f_ng_restriction', lower=0, units='N')

# Initial run boundary constraint
initialrun.add_boundary_constraint(name='mass', loc='initial', units='kg', upper=airplane.limits.MTOW)
initialrun.add_boundary_constraint(name='initial_run_eom.f_ng', loc='final', units='N',
                                   lower=0, upper=1)


initialrun.add_objective('mass', loc='initial', scaler=-1)

initialrun.add_timeseries_output(
        ['aero.*', 'initial_run_eom.f_mg', 'initial_run_eom.f_ng'],
        units={
            'aero.L': 'kN', 'aero.D': 'kN', 'aero.M': 'N*m',
            'initial_run_eom.f_mg': 'kN', 'initial_run_eom.f_ng': 'kN'
        })

p.setup()

p.set_val('traj.initialrun.t_initial', 0)
p.set_val('traj.initialrun.t_duration', 60)

p.set_val('traj.initialrun.parameters:elevation', runway.elevation)
p.set_val('traj.initialrun.parameters:rw_slope', runway.slope)

p['traj.initialrun.states:x'] = initialrun.interpolate(ys=[0, 0.8 * runway.tora],
                                                       nodes='state_input')
p['traj.initialrun.states:V'] = initialrun.interpolate(ys=[0, 120], nodes='state_input')
p['traj.initialrun.states:mass'] = initialrun.interpolate(
        ys=[airplane.limits.MTOW, airplane.limits.MTOW - 600], nodes='state_input')
p['traj.initialrun.parameters:de'] = 0.0

dm.run_problem(p)
sim_out = traj.simulate()

# Plots
plot_results([('traj.initialrun.timeseries.states:x',
               'traj.initialrun.timeseries.states:V',
               'Distance (m)',
               'Velocity (kn)'),
              ('traj.initialrun.timeseries.states:x',
               'traj.initialrun.timeseries.states:mass',
               'Distance (m)',
               'Mass (kg)')],
             title='Ground Roll',
             p_sol=p,
             p_sim=sim_out)

plt.show()
