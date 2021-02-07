import openmdao.api as om
from scipy.interpolate import interp1d
from scipy.constants import degree
import numpy as np

from toa.data import Airplane
from toa.models.aero.graphs import flap_alfadelta_x
from toa.models.aero.graphs import flap_alfadelta_y
from toa.models.aero.graphs import flap_clmax_base_x
from toa.models.aero.graphs import flap_clmax_base_y
from toa.models.aero.graphs import flap_k2_x
from toa.models.aero.graphs import flap_k2_y
from toa.models.aero.graphs import flap_k3_x
from toa.models.aero.graphs import flap_k3_y
from toa.models.aero.graphs import flap_kb_x
from toa.models.aero.graphs import flap_kb_y
from toa.models.aero.graphs import slat_clmax_base_x
from toa.models.aero.graphs import slat_clmax_base_y
from toa.models.aero.graphs import slat_leff_x
from toa.models.aero.graphs import slat_leff_y


class FlapSlatComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('airplane', types=Airplane, desc='Class containing all airplane data')

    def setup(self):
        airplane = self.options['airplane']

        self.add_input(name='flap_angle', val=0.0, desc='Flap deflection', units='deg')

        self.add_output(name='CL0', val=airplane.coeffs.CL0, desc='Lift coefficient for alpha zero', units=None)
        self.add_output(name='CLmax', val=airplane.coeffs.CLmax, desc='Max lift coefficient', units=None)
        self.add_output(name='CLa', val=airplane.coeffs.CLa, desc='Lift x alfa curve slope', units='1/rad')
        self.add_output(name='alpha_max', val=airplane.coeffs.alpha_max, desc='Max Angle of attack', units='deg')

        self.declare_partials(of='CL0', wrt='flap_angle', method='fd')
        self.declare_partials(of='CLmax', wrt='flap_angle', method='fd')
        self.declare_partials(of='CLa', wrt='flap_angle', method='fd')
        self.declare_partials(of='alpha_max', wrt='flap_angle', method='fd')

    def _calc_flap2D(self, flap_angle, airplane, clinha_c=1.075):
        alpha_delta = interp1d(flap_alfadelta_x, flap_alfadelta_y, kind='cubic')(flap_angle)
        dcl = airplane.coeffs.cla * alpha_delta * flap_angle * degree

        k1 = 0.9
        k2 = interp1d(flap_k2_x, flap_k2_y, kind='cubic')(flap_angle)
        k3 = interp1d(flap_k3_x, flap_k3_y, kind='cubic')(flap_angle / 45)
        cl_max_base = interp1d(flap_clmax_base_x, flap_clmax_base_y, kind='cubic')(airplane.wing.t_c * 100)

        dcl_max = k1 * k2 * k3 * cl_max_base
        cla_2d_flap = clinha_c * airplane.coeffs.cla
        return dcl, dcl_max, cla_2d_flap

    def _calc_flap3D(self, flap_angle, airplane, clinha_c=1.075):
        dcl, dcl_max, cla_2d_flap = self._calc_flap2D(flap_angle, airplane)
        kb = interp1d(flap_kb_x, flap_kb_y, kind='cubic')(airplane.flap.bf_b)
        dCL_flap = kb * dcl * airplane.coeffs.CLa / airplane.coeffs.cla * 1.0526
        CLalpha_flap = airplane.coeffs.CLa * (1 + (clinha_c - 1) * (airplane.flap.bf_b))
        k_lambda = (1 - 0.08 * np.cos(airplane.wing.sweep_14 * degree) ** 2) * np.cos(
                airplane.wing.sweep_14 * degree) ** (3 / 4)
        dCLmax = dcl_max * airplane.flap.bf_b * k_lambda
        return dCL_flap, CLalpha_flap, dCLmax

    def _calc_slat2D(self, slat_angle, airplane, clinha_c=1.075):
        le_eff = interp1d(slat_leff_x, slat_leff_y, kind='cubic')(airplane.slat.cs_c)
        dcl_slat = le_eff * clinha_c * slat_angle
        cl_max_base = interp1d(slat_clmax_base_x, slat_clmax_base_y, kind='cubic')(airplane.slat.cs_c)
        dcl_max_slat = cl_max_base * clinha_c * slat_angle * degree * 1.2
        return dcl_slat, dcl_max_slat

    def _calc_slat3D(self, slat_angle, airplane, clinha_c=1.075):
        dcl_slat, _ = self._calc_slat2D(slat_angle, airplane)
        kb = interp1d(flap_kb_x, flap_kb_y, kind='cubic')(airplane.slat.bs_b)
        dCL_slat = kb * dcl_slat * airplane.coeffs.CLa / airplane.coeffs.cla * 1.0526
        dCLmax_slat = 7.11 * airplane.slat.cs_c * 0.8268 ** 2 * np.cos(
                airplane.wing.sweep_14 * degree) ** 2
        return dCL_slat, dCLmax_slat

    def compute(self, inputs, outputs, **kwargs):
        flap_angle = inputs['flap_angle']
        airplane = self.options['airplane']

        if flap_angle > 0:
            slat_angle = 25
            dCL_flap, CLalpha_flap, dCLmax_flap = self._calc_flap3D(flap_angle, airplane)
            dCL_slat, dCLmax_slat = self._calc_slat3D(slat_angle, airplane)

            CL0 = airplane.coeffs.CL0 + dCL_flap + dCL_slat
            CLmax = airplane.coeffs.CLmax + dCLmax_flap + dCLmax_slat
            CLa = CLalpha_flap
            alpha_max = (CLmax - CL0) / CLalpha_flap / degree

        else:
            CL0 = airplane.coeffs.CL0
            CLmax = airplane.coeffs.CLmax
            CLa = airplane.coeffs.CLa
            alpha_max = airplane.coeffs.alpha_max

        outputs['CL0'] = CL0
        outputs['CLmax'] = CLmax
        outputs['CLa'] = CLa
        outputs['alpha_max'] = alpha_max
