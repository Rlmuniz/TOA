import math

from ambiance import Atmosphere


def validate_result(p, airplane, runway, flap_angle):
    ## Ground roll
    mu = 0.025
    g = 9.80665
    atmos = Atmosphere(runway.elevation)

    VR = p.get_val('traj.initial_run.timeseries.states:V')[-1]

    T0 = p.get_val('traj.initial_run.timeseries.thrust')[0]
    TR = p.get_val('traj.initial_run.timeseries.thrust')[-1]

    W0 = p.get_val('traj.initial_run.timeseries.states:mass')[0] * g
    WR = p.get_val('traj.initial_run.timeseries.states:mass')[-1] * g

    Sngr_opt = p.get_val('traj.initial_run.timeseries.states:x')[-1]

    CLg = p.get_val('traj.initial_run.timeseries.CLg')[-1]
    CDg = p.get_val('traj.initial_run.timeseries.CD')[-1]

    agV0 = g * (T0 / W0 - mu - runway.slope)
    agVR = g * ((TR / WR - mu) - (CDg - mu * CLg) * atmos.density[0] * VR ** 2 / (
                2 * WR / airplane.wing.area) - runway.slope)

    k = ((1 - agVR/agV0)/math.log(agV0/agVR))
    agave = k * agV0

    Sngr = VR ** 2 / (2 * agave)

    print(f"Sngr: {Sngr}, Sngr_opt: {Sngr_opt}, diff: {(Sngr_opt - Sngr)/Sngr_opt * 100}")

    ## Rotation
    Vlof = p.get_val('traj.rotation.timeseries.states:V')[-1]
    tr = p.get_val('traj.rotation.timeseries.time')[-1] - p.get_val('traj.rotation.timeseries.time')[0]
    alpha_max = p.get_val('traj.rotation.timeseries.states:theta')[-1]
    Sr = 0.5 * (VR + Vlof) * tr
    Sr_opt = p.get_val('traj.rotation.timeseries.states:x')[-1] - Sngr_opt

    print(f"Sngr: {Sr}, Sngr_opt: {Sr_opt}, diff: {(Sr_opt - Sr) / Sr_opt * 100}, alpha_max: {alpha_max}, tr: {tr}")
    p.get_val('traj.transition.timeseries.time')[-1] - p.get_val('traj.rotation.timeseries.time')[-1]
    ## Transition
    if flap_angle == 0:
        CLmax = 1.4156
    elif flap_angle == 5:
        CLmax = 2.0581687373857513
    elif flap_angle == 10:
        CLmax = 2.1299732557370072
    else:
        CLmax = 2.2234101823709764

    vlof_vs = p.get_val('traj.transition.timeseries.V_Vstall')[0]

    deltaCL = 0.5 * (vlof_vs ** 2 - 1) * (CLmax * (vlof_vs ** 2 - 0.53) + 0.38)
    W = p.get_val('traj.rotation.timeseries.states:mass')[-1] * g
    Rtr = 2 * (W / airplane.wing.area) / (atmos.density[0] * g * deltaCL)
    T = p.get_val('traj.rotation.timeseries.thrust')[-1]
    D = p.get_val('traj.rotation.timeseries.D')[-1]
    theta = (T - D)/W

    Str = Rtr * math.sin(theta)

    htr = Str * theta/2
    hscreen = p.get_val('traj.transition.timeseries.h_mlg', units='m')[-1]
    if htr > hscreen:
        Scl = 0
    if htr < hscreen:
        Scl = (hscreen - htr)/math.tan(theta)

    Sa = Str + Scl

    Sroskam = Sngr + Sr + Sa
    Sopt = p.get_val('traj.transition.timeseries.x_mlg')[-1]
    print(f"Sroskam: {Sroskam}, Programa: {Sopt}, diff: {(Sopt - Sroskam)/Sopt * 100}")
