import openmdao.api as om
import dymos as dm

from toa.data import get_airplane_data
from toa.ode.initialrun_ode import InitialRunODE
from toa.ode.rotation_ode import RotationODE
from toa.ode.transition_ode import TransitionODE
from toa.runway import Runway


def run_takeoff(airplane, runway, flap_angle=0.0, wind_speed=0.0):
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    # p.driver.declare_coloring()

    p.model.linear_solver = om.DirectSolver()

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    # --------------------------------------------- Initial Run --------------------------------------------------------
    initial_run = dm.Phase(ode_class=InitialRunODE,
                           transcription=dm.GaussLobatto(num_segments=20, compressed=False),
                           ode_init_kwargs={'airplane': airplane})

    traj.add_phase('initial_run', initial_run)

    initial_run.set_time_options(fix_initial=True, units='s', duration_bounds=(10, 100))

    # Initial run states
    initial_run.add_state(name='V', units='m/s', rate_source='initial_run_eom.v_dot', targets=['V'], fix_initial=True,
                          fix_final=False, lower=0, ref=100, defect_ref=100)
    initial_run.add_state(name='x', units='m', rate_source='initial_run_eom.x_dot',
                          targets=['mlg_pos.x'], fix_initial=True, fix_final=False,
                          lower=airplane.landing_gear.main.x, ref=1000, defect_ref=1000)
    initial_run.add_state(name='mass', units='kg', rate_source='prop.m_dot',
                          targets=['mass'], fix_initial=False, fix_final=False, lower=0.0,
                          upper=airplane.limits.MTOW, ref=10000, defect_ref=10000)

    # Initial run parameters
    # initial_run.add_parameter(name='de', val=0.0, units='deg', desc='Elevator deflection',
    #                          targets=['aero.de'], opt=False, include_timeseries=True)
    initial_run.add_parameter(name='theta', val=0.0, units='deg', desc='Pitch Angle',
                              targets=['aero.alpha', 'initial_run_eom.alpha',
                                       'mlg_pos.theta'], opt=False, include_timeseries=True)
    initial_run.add_parameter(name='h', val=0.0, units='m',
                              desc='Vertical CG position',
                              targets=['mlg_pos.h', 'aero.ground_effect.h'], opt=False, include_timeseries=True)

    # path constraint
    initial_run.add_path_constraint(name='initial_run_eom.f_mg', lower=0, units='N')
    initial_run.add_path_constraint(name='initial_run_eom.f_ng', lower=0, units='N')

    initial_run.add_boundary_constraint(name='initial_run_eom.f_ng', loc='final', units='N', lower=0.0, upper=0.2,
                                        shape=(1,))

    initial_run.add_control(name='de', units='deg', lower=-20.0, upper=20.0, targets=['aero.de'], rate_continuity=True,
                            ref=10)

    # initial_run.add_objective('mass', loc='initial', scaler=-1)

    initial_run.add_timeseries_output('initial_run_eom.f_mg', units='kN')
    initial_run.add_timeseries_output('initial_run_eom.f_ng', units='kN')
    initial_run.add_timeseries_output('initial_run_eom.v_dot')
    initial_run.add_timeseries_output('aero.CL')
    initial_run.add_timeseries_output('aero.cl_ground_corr.CLg')
    initial_run.add_timeseries_output('aero.CD')
    initial_run.add_timeseries_output('aero.Cm')
    initial_run.add_timeseries_output('prop.thrust')
    initial_run.add_timeseries_output('mlg_pos.x_mlg')
    initial_run.add_timeseries_output('mlg_pos.h_mlg', units='ft')
    initial_run.add_timeseries_output('tas_comp.tas')

    # --------------------------------------------- Rotation -----------------------------------------------------------
    rotation = dm.Phase(ode_class=RotationODE,
                        transcription=dm.GaussLobatto(num_segments=10, compressed=False),
                        ode_init_kwargs={'airplane': airplane})
    traj.add_phase(name='rotation', phase=rotation)

    rotation.set_time_options(fix_initial=False, units='s', duration_bounds=(1, 20))

    # Rotation states
    rotation.add_state(name='V', units='m/s', rate_source='rotation_eom.v_dot',
                       targets=['V'], fix_initial=False, fix_final=False, lower=0, ref=100, defect_ref=100)
    rotation.add_state(name='x', units='m', rate_source='rotation_eom.x_dot',
                       targets=['mlg_pos.x'], fix_initial=False, fix_final=False, lower=0, ref=1000,
                       defect_ref=1000)
    rotation.add_state(name='h', units='m', rate_source='rotation_eom.h_dot',
                       targets=['mlg_pos.h', 'aero.ground_effect.h'], lower=airplane.landing_gear.main.z,
                       fix_initial=True, fix_final=False, ref=10,
                       defect_ref=10)
    rotation.add_state(name='mass', units='kg', rate_source='prop.m_dot', targets=['mass'],
                       fix_initial=False, fix_final=False, lower=0.0, ref=10000, defect_ref=10000)
    rotation.add_state(name='theta', units='deg', rate_source='rotation_eom.theta_dot',
                       targets=['aero.alpha', 'rotation_eom.alpha', 'mlg_pos.theta'],
                       fix_initial=True, fix_final=False, lower=0.0, ref=10, defect_ref=10)
    rotation.add_state(name='q', units='deg/s', rate_source='rotation_eom.q_dot',
                       targets=['q'], fix_initial=True, fix_final=False, lower=0.0, ref=10, defect_ref=10)

    # Rotation controls
    rotation.add_control(name='de', units='deg', lower=-20.0, upper=20.0, targets=['aero.de'], rate_continuity=True)

    # Rotation path constraints
    rotation.add_path_constraint(name='rotation_eom.f_mg', lower=0, units='N')

    # Rotation boundary constraint
    rotation.add_boundary_constraint(name='rotation_eom.f_mg', loc='final', units='N', lower=0.0, upper=0.2, shape=(1,))
    rotation.add_boundary_constraint(name='V', loc='final', units='m/s', upper=93, shape=(1,))

    rotation.add_timeseries_output('rotation_eom.f_mg', units='kN')
    rotation.add_timeseries_output('rotation_eom.h_dot', units='ft/min')
    rotation.add_timeseries_output('rotation_eom.v_dot')
    rotation.add_timeseries_output('aero.CL')
    rotation.add_timeseries_output('aero.cl_ground_corr.CLg')
    rotation.add_timeseries_output('aero.CD')
    rotation.add_timeseries_output('aero.Cm')
    rotation.add_timeseries_output('aero.D')
    rotation.add_timeseries_output('prop.thrust')
    rotation.add_timeseries_output('mlg_pos.x_mlg')
    rotation.add_timeseries_output('mlg_pos.h_mlg', units='ft')
    rotation.add_timeseries_output('tas_comp.tas')
    # --------------------------------------------- Transition ---------------------------------------------------------
    transition = dm.Phase(ode_class=TransitionODE, transcription=dm.GaussLobatto(num_segments=10, compressed=False),
                          ode_init_kwargs={'airplane': airplane})
    traj.add_phase(name='transition', phase=transition)

    transition.set_time_options(fix_initial=False, units='s')

    # states
    transition.add_state(name='V', units='m/s', rate_source='transition_eom.v_dot',
                         targets=['V'], fix_initial=False, fix_final=False, lower=0, ref=100, defect_ref=100)
    transition.add_state(name='x', units='m', rate_source='transition_eom.x_dot',
                         targets=['mlg_pos.x', 'obj_cmp.x'], fix_initial=False, fix_final=False, lower=0, ref=1000,
                         defect_ref=1000)
    transition.add_state(name='h', units='m', rate_source='transition_eom.h_dot',
                         targets=['mlg_pos.h', 'aero.ground_effect.h'], lower=0.0, fix_initial=False, fix_final=False,
                         ref=10,
                         defect_ref=10)
    transition.add_state(name='mass', units='kg', rate_source='prop.m_dot', targets=['mass'],
                         fix_initial=False, fix_final=False, lower=0.0, ref=10000, defect_ref=10000)
    transition.add_state(name='theta', units='deg', rate_source='transition_eom.theta_dot',
                         targets=['theta'],
                         fix_initial=False, fix_final=False, lower=0.0, ref=10, defect_ref=10)
    transition.add_state(name='gam', units='deg', rate_source='transition_eom.gam_dot',
                         targets=['gam'],
                         fix_initial=True, fix_final=False, lower=0.0, ref=10, defect_ref=10)
    transition.add_state(name='q', units='deg/s', rate_source='transition_eom.q_dot',
                         targets=['q'], fix_initial=False, fix_final=False, lower=0.0, ref=10, defect_ref=10)

    # controls
    transition.add_control(name='de', units='deg', lower=-20.0, upper=20.0, targets=['aero.de'], rate_continuity=True,
                           ref=10)

    # path constraints
    transition.add_path_constraint(name='aero.alpha_lim.alphadiff', lower=0.0, units='rad')

    # Boundary Constraint
    transition.add_boundary_constraint(name='mlg_pos.x_mlg', loc='final', units='m', upper=runway.toda, shape=(1,))
    transition.add_boundary_constraint(name='mlg_pos.h_mlg', loc='final', units='ft', equals=35.0, shape=(1,))
    transition.add_boundary_constraint(name='v_vs_comp.V_Vstall', loc='final', units=None, lower=1.13, shape=(1,))
    # transition.add_boundary_constraint(name='transition_eom.gam_dot', loc='final', units='rad/s', equals=0.0, shape=(1,))

    transition.add_objective('obj_cmp.obj', loc='final')

    transition.add_timeseries_output('transition_eom.h_dot', units='ft/min')
    transition.add_timeseries_output('transition_eom.v_dot')
    transition.add_timeseries_output('transition_eom.x_dot')
    transition.add_timeseries_output('alpha_comp.alpha')
    transition.add_timeseries_output('prop.thrust')
    transition.add_timeseries_output('mlg_pos.x_mlg')
    transition.add_timeseries_output('mlg_pos.h_mlg', units='ft')
    transition.add_timeseries_output('tas_comp.tas')
    transition.add_timeseries_output('v_vs_comp.V_Vstall')

    # ---------------------------------------- Trajectory Parameters ---------------------------------------------------
    traj.add_parameter(name='dih', val=0.0, units='deg', lower=-5.0, upper=5.0,
                       desc='Horizontal stabilizer angle',
                       targets={
                           'initial_run': ['aero.dih'],
                           'rotation': ['aero.dih'],
                           'transition': ['aero.dih']
                           },
                       opt=True, dynamic=False)
    traj.add_parameter(name='Vw', val=0.0, units='m/s',
                       desc='Wind speed along the runway, defined as positive for a headwind',
                       targets={
                           'initial_run': ['Vw'],
                           'rotation': ['Vw'],
                           'transition': ['Vw']
                           },
                       opt=False)
    traj.add_parameter(name='flap_angle', val=0.0, units='deg', desc='Flap defletion',
                       targets={
                           'initial_run': ['aero.flap_angle'],
                           'rotation': ['aero.flap_angle'],
                           'transition': ['aero.flap_angle'],
                           },
                       opt=False, dynamic=False)
    traj.add_parameter(name='elevation', val=0.0, units='m', desc='Runway elevation',
                       targets={
                           'initial_run': ['elevation'],
                           'rotation': ['elevation'],
                           'transition': ['elevation'],
                           },
                       opt=False, dynamic=False)
    traj.add_parameter(name='rw_slope', val=0.0, units='rad', desc='Runway slope',
                       targets={
                           'initial_run': ['initial_run_eom.rw_slope'],
                           'rotation': ['rotation_eom.rw_slope'],
                           },
                       opt=False, dynamic=False)

    # ------------------------------------------------ Link Phases -----------------------------------------------------
    traj.link_phases(phases=['initial_run', 'rotation'], vars=['time', 'V', 'x', 'mass', 'de'])
    traj.link_phases(phases=['rotation', 'transition'], vars=['time', 'V', 'x', 'mass', 'h', 'theta', 'q', 'de'])

    p.setup(check=True)

    p.set_val('traj.initial_run.t_initial', 0)
    p.set_val('traj.initial_run.t_duration', 60)
    p.set_val('traj.rotation.t_initial', 60)
    p.set_val('traj.rotation.t_duration', 3)
    p.set_val('traj.transition.t_initial', 63)
    p.set_val('traj.transition.t_duration', 10)

    p.set_val('traj.parameters:elevation', runway.elevation)
    p.set_val('traj.parameters:rw_slope', runway.slope)
    p.set_val('traj.parameters:flap_angle', flap_angle)
    p.set_val('traj.parameters:dih', 0.0)
    p['traj.parameters:Vw'] = wind_speed

    p['traj.initial_run.states:x'] = initial_run.interpolate(
            ys=[airplane.landing_gear.main.x, 0.7 * runway.tora],
            nodes='state_input')
    p['traj.initial_run.states:V'] = initial_run.interpolate(ys=[0, 60], nodes='state_input')
    p['traj.initial_run.states:mass'] = initial_run.interpolate(
            ys=[airplane.limits.MTOW, airplane.limits.MTOW - 100], nodes='state_input')
    p['traj.initial_run.parameters:h'] = airplane.landing_gear.main.z

    p['traj.rotation.states:x'] = rotation.interpolate(
            ys=[0.7 * runway.tora, 0.8 * runway.tora],
            nodes='state_input')
    p['traj.rotation.states:V'] = rotation.interpolate(ys=[60, 70], nodes='state_input')
    p['traj.rotation.states:mass'] = rotation.interpolate(
            ys=[airplane.limits.MTOW - 100, airplane.limits.MTOW - 200],
            nodes='state_input')
    p['traj.rotation.states:h'] = airplane.landing_gear.main.z
    p['traj.rotation.states:q'] = rotation.interpolate(ys=[0.0, 10.0], nodes='state_input')
    p['traj.rotation.states:theta'] = rotation.interpolate(ys=[0.0, 10.0], nodes='state_input')
    p['traj.rotation.controls:de'] = rotation.interpolate(ys=[0.0, -20.0], nodes='control_input')

    p['traj.transition.states:x'] = transition.interpolate(
            ys=[0.8 * runway.tora, runway.toda],
            nodes='state_input')
    p['traj.transition.states:V'] = transition.interpolate(ys=[70, 80], nodes='state_input')
    p['traj.transition.states:mass'] = transition.interpolate(
            ys=[airplane.limits.MTOW - 200, airplane.limits.MTOW - 300],
            nodes='state_input')
    p['traj.transition.states:h'] = transition.interpolate(ys=[airplane.landing_gear.main.z, 35 * 0.3048],
                                                           nodes='state_input')
    p['traj.transition.states:q'] = transition.interpolate(ys=[10.0, 5.0], nodes='state_input')
    p['traj.transition.states:theta'] = transition.interpolate(ys=[10.0, 12.0], nodes='state_input')
    p['traj.transition.states:gam'] = transition.interpolate(ys=[0.0, 5.0], nodes='state_input')

    dm.run_problem(p)
    sim_out = traj.simulate()

    print(f"RTOW: {p.get_val('traj.initial_run.timeseries.states:mass', units='kg')[0]} kg")
    print(f"Rotation speed (VR): {p.get_val('traj.initial_run.timeseries.states:V', units='kn')[-1]} kn")
    print(f"Vlof speed (Vlof): {p.get_val('traj.rotation.timeseries.states:V', units='kn')[-1]} kn")
    print(f"V3 speed (V3): {p.get_val('traj.transition.timeseries.states:V', units='kn')[-1]} kn")
    print(f"Horizontal stabilizer: {p.get_val('traj.parameters:dih')} deg")

    return p, sim_out


if __name__ == '__main__':
    runway = Runway(3000, 0.0, 0.0, 0.0, 0.0)
    airplane = get_airplane_data('b734')

    sol, sim = run_takeoff(airplane, runway)
