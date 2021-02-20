import numpy as np
import openmdao.api as om


class TrueAirspeedCompGroundRoll(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input(name='Vw', val=np.zeros(nn),
                       desc='Wind speed along the runway, defined as positive for a headwind',
                       units='m/s')
        self.add_input(name='V', val=np.zeros(nn), desc='Body x axis velocity', units='m/s')

        self.add_output(name='tas', val=np.zeros(nn), desc="True airspeed", units='m/s')

        self.declare_partials(of='tas', wrt='V', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='tas', wrt='Vw', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs, **kwargs):
        outputs['tas'] = inputs['V'] + inputs['Vw']


class TrueAirspeedComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)
        ones = np.ones(nn)

        self.add_input(name='V', val=ones, desc='Body x axis velocity', units='m/s')
        self.add_input(name='Vw', val=zz,
                       desc='Wind speed along the runway, defined as positive for a headwind',
                       units='m/s')
        self.add_input(name='gam', val=ones, desc='Flight path angle', units='rad')

        self.add_output(name='tas', val=ones, desc="True airspeed", units='m/s')

        self.declare_partials(of='tas', wrt='V', rows=ar, cols=ar)
        self.declare_partials(of='tas', wrt='Vw', rows=ar, cols=ar)
        self.declare_partials(of='tas', wrt='gam', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        V = inputs['V']
        Vw = inputs['Vw']
        gam = inputs['gam']

        cosgam = np.cos(gam)
        singam = np.sin(gam)

        Vx = V * cosgam + Vw
        Vy = V * singam

        outputs['tas'] = (Vx ** 2 + Vy ** 2) ** 0.5

    def compute_partials(self, inputs, partials, **kwargs):
        V = inputs['V']
        gam = inputs['gam']
        Vw = inputs['Vw']

        cosgam = np.cos(gam)
        singam = np.sin(gam)

        partials['tas', 'V'] = 1.0 * (V + Vw * cosgam) * (V ** 2 + 2 * V * Vw * cosgam + Vw ** 2) ** (-0.5)
        partials['tas', 'gam'] = -1.0 * V * Vw * (V ** 2 + 2 * V * Vw * cosgam + Vw ** 2) ** (-0.5) * singam
        partials['tas', 'Vw'] = 1.0 * (V * cosgam + Vw) * (V ** 2 + 2 * V * Vw * cosgam + Vw ** 2) ** (-0.5)


if __name__ == '__main__':
    prob = om.Problem()
    num_nodes = 20
    prob.model.add_subsystem('comp', TrueAirspeedComp(num_nodes=num_nodes))

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    prob.check_partials(compact_print=True, show_only_incorrect=True)
