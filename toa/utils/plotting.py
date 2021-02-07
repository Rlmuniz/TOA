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
    roc_sol = {
        'initial_run': p_sol.get_val('traj.initial_run.timeseries.parameters:de'),
        'rotation': p_sol.get_val('traj.rotation.timeseries.h_dot'),
        'transition': p_sol.get_val('traj.transition.timeseries.h_dot'),
        }
    roc_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.parameters:de'),
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
        'rotation': p_sol.get_val('traj.rotation.timeseries.states:theta'),
        'transition': p_sol.get_val('traj.transition.timeseries.alpha')
        }
    alpha_sim = {
        'rotation': p_sim.get_val('traj.rotation.timeseries.states:theta'),
        'transition': p_sim.get_val('traj.transition.timeseries.alpha')
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
        'rotation': p_sol.get_val('traj.rotation.timeseries.controls:de'),
        'transition': p_sol.get_val('traj.transition.timeseries.controls:de'),
        }
    de_sim = {
        'initial_run': p_sim.get_val('traj.initial_run.timeseries.parameters:de'),
        'rotation': p_sim.get_val('traj.rotation.timeseries.controls:de'),
        'transition': p_sim.get_val('traj.transition.timeseries.controls:de'),
        }

    # Plots
    # Distancias
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    axs[0].plot(time_sim['initial_run'], x_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-', label='Simulacao')
    axs[0].plot(time_sim['rotation'], x_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[0].plot(time_sim['transition'], x_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[0].plot(time_sol['initial_run'], x_sol['initial_run'], marker='o', color='tab:blue', linestyle='None', label='Colocacao')
    axs[0].plot(time_sol['rotation'], x_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[0].plot(time_sol['transition'], x_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[0].set_xlabel('Tempo (s)')
    axs[0].set_ylabel('Horizontal (m)')
    axs[0].grid(True)

    axs[1].plot(time_sim['initial_run'], h_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-')
    axs[1].plot(time_sim['rotation'], h_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[1].plot(time_sim['transition'], h_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[1].plot(time_sol['initial_run'], h_sol['initial_run'], marker='o', color='tab:blue', linestyle='None')
    axs[1].plot(time_sol['rotation'], h_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[1].plot(time_sol['transition'], h_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[1].set_xlabel('Tempo (s)')
    axs[1].grid(True)
    axs[1].set_ylabel('Vertical (ft)')
    axs[1].set_ylim(bottom=0, top=50)
    fig.legend(loc='lower center', ncol=2)
    plt.savefig(os.path.join(plots_dir, 'deslocamentos.png'))

    # Velocidade e Tracao e Massa
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))
    axs[0].plot(time_sim['initial_run'], v_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-',
                label='Simulacao')
    axs[0].plot(time_sim['rotation'], v_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[0].plot(time_sim['transition'], v_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[0].plot(time_sol['initial_run'], v_sol['initial_run'], marker='o', color='tab:blue', linestyle='None',
                label='Colocacao')
    axs[0].plot(time_sol['rotation'], v_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[0].plot(time_sol['transition'], v_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[0].set_xlabel('Tempo (s)')
    axs[0].set_ylabel('Velocidade (kt)')
    axs[0].grid(True)

    axs[1].plot(time_sim['initial_run'], roc_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-')
    axs[1].plot(time_sim['rotation'], roc_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[1].plot(time_sim['transition'], roc_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[1].plot(time_sol['initial_run'], roc_sol['initial_run'], marker='o', color='tab:blue', linestyle='None')
    axs[1].plot(time_sol['rotation'], roc_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[1].plot(time_sol['transition'], roc_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[1].set_xlabel('Tempo (s)')
    axs[1].set_ylabel('Razao de subida (ft/min)')
    axs[1].grid(True)

    axs[2].plot(time_sim['initial_run'], mass_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-')
    axs[2].plot(time_sim['rotation'], mass_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[2].plot(time_sim['transition'], mass_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[2].plot(time_sol['initial_run'], mass_sol['initial_run'], marker='o', color='tab:blue', linestyle='None')
    axs[2].plot(time_sol['rotation'], mass_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[2].plot(time_sol['transition'], mass_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[2].set_xlabel('Tempo (s)')
    axs[2].set_ylabel('Massa (kg)')
    axs[2].grid(True)

    axs[3].plot(time_sim['initial_run'], thrust_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-')
    axs[3].plot(time_sim['rotation'], thrust_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[3].plot(time_sim['transition'], thrust_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[3].plot(time_sol['initial_run'], thrust_sol['initial_run'], marker='o', color='tab:blue', linestyle='None')
    axs[3].plot(time_sol['rotation'], thrust_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[3].plot(time_sol['transition'], thrust_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[3].set_xlabel('Tempo (s)')
    axs[3].set_ylabel('Tracao (kN)')
    axs[3].grid(True)

    fig.legend(loc='lower center', ncol=2)
    plt.savefig(os.path.join(plots_dir, 'velocidade_e_massa.png'))

    # Angulos
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
    axs[0].plot(time_sim['initial_run'], theta_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-',
                label='Simulacao')
    axs[0].plot(time_sim['rotation'], theta_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[0].plot(time_sim['transition'], theta_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[0].plot(time_sol['initial_run'], theta_sol['initial_run'], marker='o', color='tab:blue', linestyle='None',
                label='Colocacao')
    axs[0].plot(time_sol['rotation'], theta_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[0].plot(time_sol['transition'], theta_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[0].set_xlabel('Tempo (s)')
    axs[0].set_ylabel('Arfagem (deg)')
    axs[0].grid(True)

    axs[1].plot(time_sim['transition'], gam_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[1].plot(time_sol['transition'], gam_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[1].set_xlabel('Tempo (s)')
    axs[1].set_ylabel('Trajetoria (deg)')
    axs[1].grid(True)

    axs[2].plot(time_sim['initial_run'], de_sim['initial_run'], color='tab:orange', linewidth=2, linestyle='-')
    axs[2].plot(time_sim['rotation'], de_sim['rotation'], color='tab:orange', linewidth=2, linestyle='-')
    axs[2].plot(time_sim['transition'], de_sim['transition'], color='tab:orange', linewidth=2, linestyle='-')

    axs[2].plot(time_sol['initial_run'], de_sol['initial_run'], marker='o', color='tab:blue', linestyle='None')
    axs[2].plot(time_sol['rotation'], de_sol['rotation'], marker='o', color='tab:blue', linestyle='None')
    axs[2].plot(time_sol['transition'], de_sol['transition'], marker='o', color='tab:blue', linestyle='None')
    axs[2].set_xlabel('Tempo (s)')
    axs[2].set_ylabel('Profundor (deg)')
    axs[2].grid(True)

    fig.legend(loc='lower center', ncol=2)
    plt.savefig(os.path.join(plots_dir, 'angulos.png'))