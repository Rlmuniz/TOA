import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from dymos.examples.plotting import plot_results

from toa.data import get_airplane_data
from toa.ode.initialrun_ode import InitialRunODE
from toa.ode.rotation_ode import RotationODE
from toa.ode.transition_ode import TransitionODE

from toa.runway import Runway

p = om.Problem(model=om.Group())

# set driver
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SLSQP'

runway = Runway(2500, 0.0, 0.0, 0.0, 0.0)
airplane = get_airplane_data('b734')

traj = p.model.add_subsystem('traj', dm.Trajectory())
initialrun = dm.Phase(ode_class=InitialRunODE,
                      transcription=dm.GaussLobatto(num_segments=25, compressed=False),
                      ode_init_kwargs={'airplane': airplane})
traj.add_phase(name='initialrun', phase=initialrun)

initialrun.set_time_options(fix_initial=True, units='s')

# Initial run states
initialrun.add_state(name='V', units='kn', rate_source='initial_run_eom.v_dot',
                     targets=['V'], fix_initial=True, fix_final=False, lower=0)
initialrun.add_state(name='x', units='m', rate_source='initial_run_eom.x_dot',
                     targets=['mlg_pos.x'], fix_initial=True, fix_final=False,
                     lower=airplane.landing_gear.main.x)
initialrun.add_state(name='mass', units='kg', rate_source='prop.m_dot',
                     targets=['mass'], fix_initial=False, fix_final=False, lower=0.0,
                     upper=airplane.limits.MTOW)

# Initial run parameters
initialrun.add_parameter(name='de', val=0.0, units='deg', desc='Elevator deflection',
                         targets=['aero.de'], opt=False)
initialrun.add_parameter(name='theta', val=0.0, units='deg', desc='Pitch Angle',
                         targets=['aero.alpha', 'initial_run_eom.alpha',
                                  'mlg_pos.theta'], opt=False)
initialrun.add_parameter(name='h', val=0.0, units='m',
                         desc='Vertical CG position',
                         targets=['mlg_pos.h'], opt=False)

# Initial run path constraint
initialrun.add_path_constraint(name='initial_run_eom.f_mg', lower=0, units='N')
initialrun.add_path_constraint(name='initial_run_eom.f_ng', lower=0, units='N')

# Initial run boundary constraint
initialrun.add_boundary_constraint(name='initial_run_eom.f_ng', loc='final', units='N',
                                   lower=0.0, upper=0.5)

initialrun.add_objective('mass', loc='initial', scaler=-1)

initialrun.add_timeseries_output('aero.CD')
initialrun.add_timeseries_output('aero.CL')
initialrun.add_timeseries_output('aero.L', units='kN')
initialrun.add_timeseries_output('aero.Cm')
initialrun.add_timeseries_output('theta')
initialrun.add_timeseries_output('initial_run_eom.f_mg', units='kN')
initialrun.add_timeseries_output('initial_run_eom.f_ng', units='kN')
initialrun.add_timeseries_output('prop.thrust', units='kN')
initialrun.add_timeseries_output('prop.m_dot', units='kg/s')

# Rotation
rotation = dm.Phase(ode_class=RotationODE,
                    transcription=dm.GaussLobatto(num_segments=10, compressed=False),
                    ode_init_kwargs={'airplane': airplane})
traj.add_phase(name='rotation', phase=rotation)

rotation.set_time_options(fix_initial=False, units='s')

# Rotation states
rotation.add_state(name='V', units='kn', rate_source='rotation_eom.v_dot',
                   targets=['V'], fix_initial=False, fix_final=False, lower=0)
rotation.add_state(name='x', units='m', rate_source='rotation_eom.x_dot',
                   targets=['mlg_pos.x'], fix_initial=False, fix_final=False, lower=0)
rotation.add_state(name='h', units='m', rate_source='rotation_eom.h_dot',
                   targets=['mlg_pos.h'], fix_initial=True, fix_final=False)
rotation.add_state(name='mass', units='kg', rate_source='prop.m_dot', targets=['mass'],
                   fix_initial=False, fix_final=False, lower=0.0)
rotation.add_state(name='theta', units='deg', rate_source='rotation_eom.theta_dot',
                   targets=['aero.alpha', 'rotation_eom.alpha', 'mlg_pos.theta'],
                   fix_initial=True, fix_final=False, lower=0.0)
rotation.add_state(name='q', units='deg/s', rate_source='rotation_eom.q_dot',
                   targets=['q'], fix_initial=True, fix_final=False, lower=0.0)

# Rotation controls
rotation.add_control(name='de', units='deg', lower=-30.0, upper=30.0, targets=['aero.de'], fix_initial=True, rate_continuity=True)

# Rotation path constraints
rotation.add_path_constraint(name='rotation_eom.f_mg', lower=0, units='N')

# Rotation boundary constraint
rotation.add_boundary_constraint(name='rotation_eom.f_mg', loc='final', units='N',
                                 lower=0.0, upper=0.5)

rotation.add_timeseries_output('aero.CD')
rotation.add_timeseries_output('aero.CL')
rotation.add_timeseries_output('aero.L', units='kN')
rotation.add_timeseries_output('aero.Cm')
rotation.add_timeseries_output('rotation_eom.f_mg', units='kN')
rotation.add_timeseries_output('prop.thrust', units='kN')
rotation.add_timeseries_output('prop.m_dot', units='kg/s')

# ------------------------------------------ Climb out to Screen Height ------------------------------------------------
climbout = dm.Phase(ode_class=TransitionODE, transcription=dm.GaussLobatto(num_segments=10, compressed=False),
                    ode_init_kwargs={'airplane': airplane})
traj.add_phase(name='climbout', phase=climbout)

climbout.set_time_options(fix_initial=False, units='s')

# states
climbout.add_state(name='V', units='kn', rate_source='transition_eom.v_dot',
                   targets=['V'], fix_initial=False, fix_final=False, lower=0)
climbout.add_state(name='x', units='m', rate_source='transition_eom.x_dot',
                   targets=['mlg_pos.x'], fix_initial=False, fix_final=False, lower=0)
climbout.add_state(name='h', units='m', rate_source='transition_eom.h_dot',
                   targets=['mlg_pos.h'], fix_initial=False, fix_final=False)
climbout.add_state(name='mass', units='kg', rate_source='prop.m_dot', targets=['mass'],
                   fix_initial=False, fix_final=False, lower=0.0)
climbout.add_state(name='theta', units='deg', rate_source='transition_eom.theta_dot',
                   targets=['theta'],
                   fix_initial=False, fix_final=False, lower=0.0)
climbout.add_state(name='gam', units='deg', rate_source='transition_eom.gam_dot',
                   targets=['gam'],
                   fix_initial=True, fix_final=False, lower=0.0)
climbout.add_state(name='q', units='deg/s', rate_source='transition_eom.q_dot',
                   targets=['q'], fix_initial=False, fix_final=False, lower=0.0)

# controls
climbout.add_control(name='de', units='deg', lower=-30.0, upper=30.0, targets=['aero.de'], rate_continuity=True)

# path constraints


# Boundary Constraint
climbout.add_boundary_constraint(name='mlg_pos.x_mlg', loc='final', units='m', upper=runway.toda)
climbout.add_boundary_constraint(name='mlg_pos.h_mlg', loc='final', units='ft', equals=35.0)
climbout.add_boundary_constraint(name='v_vs_comp.V_Vstall', loc='final', units=None, lower=1.2)
climbout.add_boundary_constraint(name='transition_eom.gam_dot', loc='final', units='rad/s', equals=0.0)

# Timeseries
climbout.add_timeseries_output('aero.CD')
climbout.add_timeseries_output('aero.CL')
climbout.add_timeseries_output('aero.L', units='kN')
climbout.add_timeseries_output('aero.Cm')
climbout.add_timeseries_output('prop.thrust', units='kN')
climbout.add_timeseries_output('prop.m_dot', units='kg/s')
# ---------------------------------------- Trajectory Parameters ------------------------------------------------------
# Trajectory parameters
traj.add_parameter(name='dih', val=-2.0, units='deg', lower=-10.0, upper=10.0,
                   desc='Horizontal stabilizer angle',
                   targets={
                       'initialrun': ['aero.dih'],
                       'rotation': ['aero.dih'],
                       'climbout': ['aero.dih']
                       },
                   opt=False)
traj.add_parameter(name='Vw', val=0.0, units='m/s',
                   desc='Wind speed along the runway, defined as positive for a headwind',
                   targets={
                       'initialrun': ['Vw'],
                       'rotation': ['Vw'],
                       'climbout': ['Vw']
                       },
                   opt=False, dynamic=False)
traj.add_parameter(name='flap_angle', val=0.0, units='deg', desc='Flap defletion',
                   targets={
                       'initialrun': ['aero.flap_angle'],
                       'rotation': ['aero.flap_angle'],
                       'climbout': ['aero.flap_angle'],
                       },
                   opt=False, dynamic=False)
traj.add_parameter(name='elevation', val=0.0, units='m', desc='Runway elevation',
                   targets={
                       'initialrun': ['elevation'],
                       'rotation': ['elevation'],
                       'climbout': ['elevation'],
                       },
                   opt=False, dynamic=False)
traj.add_parameter(name='rw_slope', val=0.0, units='rad', desc='Runway slope',
                   targets={
                       'initialrun': ['initial_run_eom.rw_slope'],
                       'rotation': ['rotation_eom.rw_slope'],
                       },
                   opt=False, dynamic=False)

# -------------------------------------------------- Link Phases -------------------------------------------------------
traj.link_phases(phases=['initialrun', 'rotation'],
                 vars=['time', 'V', 'x', 'mass'])

traj.link_phases(phases=['rotation', 'climbout'], vars=['time', 'V', 'x', 'mass', 'h', 'theta', 'q'])
traj.add_linkage_constraint('rotation', 'climbout', 'de', 'de')

p.model.linear_solver = om.DirectSolver()

p.setup(check=True)

p.set_val('traj.initialrun.t_initial', 0)
p.set_val('traj.initialrun.t_duration', 60)
p.set_val('traj.rotation.t_duration', 5)
p.set_val('traj.climbout.t_duration', 10)

p.set_val('traj.parameters:elevation', runway.elevation)
p.set_val('traj.parameters:rw_slope', runway.slope)
p.set_val('traj.parameters:flap_angle', 0.0)
p.set_val('traj.parameters:dih', -2.0)
p.set_val('traj.parameters:Vw', 0.0)

p['traj.initialrun.states:x'] = initialrun.interpolate(
        ys=[airplane.landing_gear.main.x, 0.7 * runway.tora],
        nodes='state_input')
p['traj.initialrun.states:V'] = initialrun.interpolate(ys=[0, 150], nodes='state_input')
p['traj.initialrun.states:mass'] = initialrun.interpolate(
        ys=[airplane.limits.MTOW, airplane.limits.MTOW - 600], nodes='state_input')
p['traj.initialrun.parameters:h'] = airplane.landing_gear.main.z

p['traj.rotation.states:x'] = rotation.interpolate(
        ys=[0.7 * runway.tora, 0.8 * runway.tora],
        nodes='state_input')
p['traj.rotation.states:V'] = rotation.interpolate(ys=[150, 160], nodes='state_input')
p['traj.rotation.states:mass'] = rotation.interpolate(
        ys=[airplane.limits.MTOW - 600, airplane.limits.MTOW - 1000],
        nodes='state_input')
p['traj.rotation.states:h'] = airplane.landing_gear.main.z
p['traj.rotation.states:q'] = rotation.interpolate(ys=[0.0, 10.0],
                                                   nodes='state_input')
p['traj.rotation.states:theta'] = rotation.interpolate(ys=[0.0, 10.0],
                                                       nodes='state_input')
p['traj.rotation.controls:de'] = rotation.interpolate(ys=[0.0, -20.0],
                                                       nodes='control_input')

p['traj.climbout.states:x'] = climbout.interpolate(
        ys=[0.8 * runway.tora, runway.toda],
        nodes='state_input')
p['traj.climbout.states:V'] = climbout.interpolate(ys=[160, 170], nodes='state_input')
p['traj.climbout.states:mass'] = climbout.interpolate(
        ys=[airplane.limits.MTOW - 1000, airplane.limits.MTOW - 1200],
        nodes='state_input')
p['traj.climbout.states:h'] = climbout.interpolate(ys=[airplane.landing_gear.main.z, 35 * 0.3048], nodes='state_input')
p['traj.climbout.states:q'] = climbout.interpolate(ys=[10.0, 5.0],
                                                   nodes='state_input')
p['traj.climbout.states:theta'] = climbout.interpolate(ys=[10.0, 12.0],
                                                       nodes='state_input')
p['traj.climbout.states:gam'] = climbout.interpolate(ys=[0.0, 0.0], nodes='state_input')

dm.run_problem(p)
sim_out = traj.simulate()

print(f"RTOW: {p.get_val('traj.initialrun.timeseries.states:mass', units='kg')[0]} kg")
print(f"Rotation speed (VR): {p.get_val('traj.initialrun.timeseries.states:V', units='kn')[-1]} kn")
print(f"Vlof speed (Vlof): {p.get_val('traj.rotation.timeseries.states:V', units='kn')[-1]} kn")
print(f"V3 speed (V3): {p.get_val('traj.climbout.timeseries.states:V', units='kn')[-1]} kn")
print(f"Horizontal stabilizer: {p.get_val('traj.parameters:dih')} deg")


# --------------------------------------- Plots_
plot_results([('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.states:V',
               'Time (s)', 'Speed (kt)'),
              ('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.states:x',
               'time (s)', 'Position (m)'),
              ('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.states:mass',
               'time (s)', 'Mass (kg)'),
              ],
             title='Initial Run',
             p_sol=p, p_sim=sim_out)

plot_results([('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:V',
               'Time (s)', 'Speed (kt)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:x',
               'time (s)', 'Position (m)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:mass',
               'time (s)', 'Mass (kg)'),
              ],
             title='Rotation',
             p_sol=p, p_sim=sim_out)

plot_results([('traj.rotation.timeseries.time', 'traj.rotation.timeseries.polynomial_controls:de',
               'Time (s)', 'De (deg)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:q',
               'time (s)', 'q (deg/s)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:theta',
               'time (s)', 'Theta (s)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:h',
               'time (s)', 'h (m)'),
              ],
             title='Rotation',
             p_sol=p, p_sim=sim_out)

plot_results([('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.CL',
               'Time (s)', 'CL'),
              ('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.CD',
               'time (s)', 'CD'),
              ('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.Cm',
               'time (s)', 'CM'),
              ],
             title='Initial Run',
             p_sol=p, p_sim=sim_out)

plot_results([('traj.rotation.timeseries.time', 'traj.rotation.timeseries.CL',
               'Time (s)', 'CL'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.CD',
               'time (s)', 'CD'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.Cm',
               'time (s)', 'CM'),
              ],
             title='Rotation',
             p_sol=p, p_sim=sim_out)

plot_results([('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.f_mg',
               'Time (s)', 'fmg'),
              ('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.f_ng',
               'time (s)', 'fng'),
              ('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.L',
               'time (s)', 'L (kN)'),
              ],
             title='Initial Run',
             p_sol=p, p_sim=sim_out)

plot_results([('traj.rotation.timeseries.time', 'traj.rotation.timeseries.f_mg',
               'Time (s)', 'fmg'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.L',
               'time (s)', 'L (kN)'),
              ],
             title='Rotation',
             p_sol=p, p_sim=sim_out)
