from scipy.interpolate import interp1d
import numpy as np

from toa.data import Airplane
from scipy.constants import degree

# Valor conjecturado (Gudmundsson) -> corresponde ao aumento da corda do perfil devido
# a deflexao do flap e slat
from toa.data import get_airplane_data

clinha_c = 1 + 7.5 / 100

def _calc_flap_2d(dflap: float, airplane: Airplane):
    """Calculate flap effect on airfoil

    dflap (deg)
    Dados removidos de abacos do ROSKAM Part6
    """
    alpha_delta = interp1d([0, 10, 20, 30, 40], [0.373, 0.3698, 0.3547, 0.3207, 0.264],
                           kind='cubic')(dflap)
    dcl = airplane.coeffs.cla * alpha_delta * dflap * degree

    k1 = 0.9
    k2 = interp1d([0, 10, 20, 30, 40, 45], [0.2, 0.46, 0.7, 0.86, 0.96, 1.0],
                  kind='cubic')(dflap)
    k3 = interp1d([0, 0.3, 0.6, 0.8, 1.0], [0.0, 0.4, 0.7, 0.87, 1.0], kind='cubic')(
            dflap / 45)
    cl_max_base = interp1d([0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
                           [1, 1, 1.023, 1.0615, 1.123, 1.224, 1.37, 1.5467, 1.7167,
                            1.8], kind='cubic')(airplane.wing.t_c * 100)

    dcl_max = k1 * k2 * k3 * cl_max_base
    cla_2d_flap = clinha_c * airplane.coeffs.cla
    return dcl, dcl_max, cla_2d_flap

def _calc_slat_2d(dslat: float, airplane: Airplane):
    le_eff = interp1d([0, .1, .15, .2, .25], [0, 1.636e-3, 2.783e-3, 4.588e-3, 7.2195],
                      kind='cubic')(airplane.slat.cs_c)
    dcl_slat = le_eff * clinha_c * dslat
    cl_max_base = interp1d([0, .05, .1, .15, .2, .25],
                           [0, .9024, 1.2, 1.433, 1.6, 1.72], kind='cubic')(airplane.slat.cs_c)
    dcl_max_slat = cl_max_base * clinha_c * dslat * degree * 1.2
    return dcl_slat, dcl_max_slat

def calc_flap_3D(dflap: float, airplane: Airplane):
    dcl, dcl_max, cla_2d_flap = _calc_flap_2d(dflap, airplane)
    kb = interp1d([0, .2, .4, .6, .8], [0, .33, .52, .74, .91], kind='cubic')(airplane.flap.bf_b)
    dCL_flap = kb * dcl * airplane.coeffs.CLa / airplane.coeffs.cla * 1.0526
    CLalpha_flap = airplane.coeffs.CLa * (1 + (clinha_c - 1) * (airplane.flap.bf_b))
    k_lambda = (1 - 0.08 * np.cos(airplane.wing.sweep_14 * degree) ** 2) * np.cos(
            airplane.wing.sweep_14 * degree) ** (3 / 4)
    dCLmax = dcl_max * airplane.flap.bf_b * k_lambda
    return dCL_flap, CLalpha_flap, dCLmax

def calc_slat_3D(dslat: float, airplane: Airplane):
    dcl_slat, _ = _calc_slat_2d(dslat, airplane)
    kb = interp1d([0, .2, .4, .6, .8], [0, .33, .52, .74, .91], kind='cubic')(airplane.slat.bs_b)
    dCL_slat = kb * dcl_slat * airplane.coeffs.CLa / airplane.coeffs.cla * 1.0526
    dCLmax_slat = 7.11 * airplane.slat.cs_c * 0.8268 ** 2 * np.cos(airplane.wing.sweep_14 * degree) ** 2
    return dCL_slat, dCLmax_slat

def get_CLdata(dflap: float, airplane: Airplane):
    if dflap > 0:
        dslat = 25
        dCL_flap, CLalpha_flap, dCLmax_flap = calc_flap_3D(dflap, airplane)
        dCL_slat, dCLmax_slat = calc_slat_3D(dslat, airplane)

        CL0 = airplane.coeffs.CL0 + dCL_flap + dCL_slat
        CLmax = airplane.coeffs.CLmax + dCLmax_flap + dCLmax_slat
        CLa = CLalpha_flap
        alpha_max = (CLmax - CL0)/CLalpha_flap / degree
    else:
        CL0 = airplane.coeffs.CL0
        CLmax = airplane.coeffs.CLmax
        CLa = airplane.coeffs.CLa
        alpha_max = airplane.coeffs.alpha_max

    return CL0, CLmax, CLa, alpha_max

if __name__ == '__main__':
    airplane = get_airplane_data('b734')
    CL0, CLmax, CLa, alpha_max = get_CLdata(10, airplane)
    print(CL0, CLmax, CLa, alpha_max)
    CL0, CLmax, CLa, alpha_max = get_CLdata(25, airplane)
    print(CL0, CLmax, CLa, alpha_max)