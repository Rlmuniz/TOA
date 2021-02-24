import os

import matplotlib.pyplot as plt


def plot_results(p_sol, p_sim, plots_dir='plots'):
    time_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.time'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.time'),
        'transition': p_sol.get_val('traj.transition.timeseries.time')
        }
    time_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.time'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.time'),
        'transition': p_sim.get_val('traj.transition.timeseries.time')
        }

    mass_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.states:mass'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.states:mass'),
        'transition': p_sol.get_val('traj.transition.timeseries.states:mass')
        }
    mass_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.states:mass'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.states:mass'),
        'transition': p_sim.get_val('traj.transition.timeseries.states:mass')
        }

    x_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.states:x'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.states:x'),
        'transition': p_sol.get_val('traj.transition.timeseries.states:x')
        }
    x_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.states:x'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.states:x'),
        'transition': p_sim.get_val('traj.transition.timeseries.states:x')
        }

    xmlg_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.x_mlg'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.x_mlg'),
        'transition': p_sol.get_val('traj.transition.timeseries.x_mlg')
        }
    xmlg_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.x_mlg'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.x_mlg'),
        'transition': p_sim.get_val('traj.transition.timeseries.x_mlg')
        }

    h_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.parameters:h', units='ft'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.states:h', units='ft'),
        'transition': p_sol.get_val('traj.transition.timeseries.states:h', units='ft'),
        }
    h_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.parameters:h', units='ft'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.states:h', units='ft'),
        'transition': p_sim.get_val('traj.transition.timeseries.states:h', units='ft'),
        }

    hmlg_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.h_mlg', units='ft'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.h_mlg', units='ft'),
        'transition': p_sol.get_val('traj.transition.timeseries.h_mlg', units='ft'),
        }
    hmlg_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.h_mlg', units='ft'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.h_mlg', units='ft'),
        'transition': p_sim.get_val('traj.transition.timeseries.h_mlg', units='ft'),
    }

    v_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.states:V', units='kn'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.states:V', units='kn'),
        'transition': p_sol.get_val('traj.transition.timeseries.states:V', units='kn'),
        }
    v_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.states:V', units='kn'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.states:V', units='kn'),
        'transition': p_sim.get_val('traj.transition.timeseries.states:V', units='kn'),
        }

    tas_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.tas', units='kn'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.tas', units='kn'),
        'transition': p_sol.get_val('traj.transition.timeseries.tas', units='kn'),
        }
    tas_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.tas', units='kn'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.tas', units='kn'),
        'transition': p_sim.get_val('traj.transition.timeseries.tas', units='kn'),
    }

    roc_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.parameters:theta'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.h_dot'),
        'transition': p_sol.get_val('traj.transition.timeseries.h_dot'),
        }
    roc_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.parameters:theta'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.h_dot'),
        'transition': p_sim.get_val('traj.transition.timeseries.h_dot'),
        }
    fmg_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.f_mg'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.f_mg')
        }
    fmg_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.f_mg'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.f_mg')
        }

    fng_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.f_ng'),
        }
    fng_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.f_ng'),
        }

    theta_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.parameters:theta'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.states:theta'),
        'transition': p_sol.get_val('traj.transition.timeseries.states:theta'),
        }
    theta_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.parameters:theta'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.states:theta'),
        'transition': p_sim.get_val('traj.transition.timeseries.states:theta'),
        }

    gam_sol = {
        'transition': p_sol.get_val('traj.transition.timeseries.states:gam'),
        }
    gam_sim = {
        'transition': p_sim.get_val('traj.transition.timeseries.states:gam'),
        }

    alpha_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.parameters:theta'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.states:theta'),
        'transition': p_sol.get_val('traj.transition.timeseries.alpha', units='deg')
        }
    alpha_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.parameters:theta'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.states:theta'),
        'transition': p_sim.get_val('traj.transition.timeseries.alpha', units='deg')
        }

    thrust_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.thrust', units='kN'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.thrust', units='kN'),
        'transition': p_sol.get_val('traj.transition.timeseries.thrust', units='kN'),
        }

    thrust_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.thrust', units='kN'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.thrust', units='kN'),
        'transition': p_sim.get_val('traj.transition.timeseries.thrust', units='kN'),
        }

    de_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.parameters:de'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.parameters:de'),
        'transition': p_sol.get_val('traj.transition.timeseries.parameters:de'),
        }
    de_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.parameters:de'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.parameters:de'),
        'transition': p_sim.get_val('traj.transition.timeseries.parameters:de'),
        }

    # Plots
    # Distancias
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(9.3, 6.3))
    axs[0].plot(xmlg_sim['initial_run'], hmlg_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-', label='Simulação')
    axs[0].plot(xmlg_sim['rotation'], hmlg_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[0].plot(xmlg_sim['transition'], hmlg_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[0].plot(xmlg_sol['initial_run'], hmlg_sol['initial_run'], marker='o', color='tab:blue', linestyle='None', label='Colocação')
    axs[0].plot(xmlg_sol['rotation'], hmlg_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[0].plot(xmlg_sol['transition'], hmlg_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[0].set_xlabel('Distância (m)')
    axs[0].set_ylabel('Altura (ft)')
    axs[0].set_xlim(0, 2000)
    axs[0].set_ylim(top=40)
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(time_sim['initial_run'], v_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-', label='Simulação')
    axs[1].plot(time_sim['rotation'], v_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[1].plot(time_sim['transition'], v_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[1].plot(time_sol['initial_run'], v_sol['initial_run'], marker='o', color='tab:blue', linestyle='None', label='Colocação')
    axs[1].plot(time_sol['rotation'], v_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[1].plot(time_sol['transition'], v_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[1].set_xlabel('Tempo (s)')
    axs[1].set_ylabel('Velocidade (kt)')
    axs[1].set_xlim(0)
    axs[1].set_ylim(top=200)
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(time_sim['initial_run'], fmg_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-', label='Simulação')
    axs[2].plot(time_sim['rotation'], fmg_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')

    axs[2].plot(time_sol['initial_run'], fmg_sol['initial_run'], marker='o', color='tab:blue', linestyle='None', label='Colocação')
    axs[2].plot(time_sol['rotation'], fmg_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[2].set_xlabel('Tempo (s)')
    axs[2].set_ylabel('Fm (kN)')
    axs[2].set_xlim(0)
    axs[2].set_ylim(top=650)
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(time_sim['initial_run'], fng_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-', label='Simulação')

    axs[3].plot(time_sol['initial_run'], fng_sol['initial_run'], marker='o', color='tab:blue', linestyle='None', label='Colocação')
    axs[3].set_xlabel('Tempo (s)')
    axs[3].set_ylabel('Fn (kN)')
    axs[3].set_xlim(0)
    axs[3].set_ylim(top=60)
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'deslocamentos.png'))

    # Velocidade e Tracao e Massa
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9.3, 6.3))
    axs[0].plot(tas_sim['initial_run'], thrust_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-', label='Simulação')
    axs[0].plot(tas_sim['rotation'], thrust_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[0].plot(tas_sim['transition'], thrust_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[0].plot(tas_sol['initial_run'], thrust_sol['initial_run'], marker='o', color='tab:blue', linestyle='None', label='Colocação')
    axs[0].plot(tas_sol['rotation'], thrust_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[0].plot(tas_sol['transition'], thrust_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[0].set_xlabel('TAS (kt)')
    axs[0].set_ylabel('Tração (kN)')
    axs[0].set_xlim(left=0)
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(time_sim['initial_run'], mass_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-', label='Simulação')
    axs[1].plot(time_sim['rotation'], mass_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[1].plot(time_sim['transition'], mass_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[1].plot(time_sol['initial_run'], mass_sol['initial_run'], marker='o', color='tab:blue', linestyle='None', label='Colocação')
    axs[1].plot(time_sol['rotation'], mass_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[1].plot(time_sol['transition'], mass_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[1].set_xlabel('Tempo (s)')
    axs[1].set_ylabel('Massa (kg)')
    axs[1].set_xlim(0)
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(time_sim['initial_run'], de_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-', label='Simulação')
    axs[2].plot(time_sim['rotation'], de_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[2].plot(time_sim['transition'], de_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[2].plot(time_sol['initial_run'], de_sol['initial_run'], marker='o', color='tab:blue', linestyle='None', label='Colocação')
    axs[2].plot(time_sol['rotation'], de_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[2].plot(time_sol['transition'], de_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[2].set_xlabel('Tempo (s)')
    axs[2].set_ylabel('Profundor (deg)')
    axs[2].set_xlim(0)
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'velocidade_e_massa.png'))

    # Angulos
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9.3, 6.3))
    axs[0].plot(time_sim['initial_run'], theta_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-',
                label='Simulação')
    axs[0].plot(time_sim['rotation'], theta_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[0].plot(time_sim['transition'], theta_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[0].plot(time_sol['initial_run'], theta_sol['initial_run'], marker='o', color='tab:blue', linestyle='None',
                label='Colocação')
    axs[0].plot(time_sol['rotation'], theta_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[0].plot(time_sol['transition'], theta_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[0].set_xlabel('Tempo (s)')
    axs[0].set_ylabel('Arfagem (deg)')
    axs[0].set_xlim(0)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time_sim['transition'], gam_sim['transition'], color='tab:orange', linewidth=2, linestyle='-', label='Simulação')

    axs[1].plot(time_sol['transition'], gam_sol['transition'], marker='o', color='tab:blue', linestyle='None', label='Colocação')
    axs[1].set_xlabel('Tempo (s)')
    axs[1].set_ylabel('Trajetória (deg)')
    axs[1].set_xlim(0)
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(time_sim['initial_run'], alpha_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-', label='Simulação')
    axs[2].plot(time_sim['rotation'], alpha_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[2].plot(time_sim['transition'], alpha_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[2].plot(time_sol['initial_run'], alpha_sol['initial_run'], marker='o', color='tab:blue', linestyle='None', label='Colocação')
    axs[2].plot(time_sol['rotation'], alpha_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[2].plot(time_sol['transition'], alpha_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[2].set_xlabel('Tempo (s)')
    axs[2].set_ylabel('Alpha (deg)')
    axs[2].set_xlim(0)
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'angulos.png'))