import numpy as np
from matplotlib import pyplot as plt


def attitude_control_sub_problem(env, alpha_setpoint=0.0, current_psi=-10, dt_sub=10, plotting=True):
    # Reset to nominal state
    obs, _ = env.reset()

    # Arbitrary initial attitude perturbation
    env.current_psi += np.deg2rad(current_psi)  # –10° initial misalignment

    # Desired alignment (prograde)
    target_alpha = alpha_setpoint

    # Simulate one main time step (600 s) via its 60 sub-steps
    num_sub_steps = env.N_sub  # 60
    dt_sub = dt_sub  # env.dt_sub = 10 s

    # Data storage
    times = [0.0]
    alphas = []
    psis = []
    cum_fuel_rot = [0.0]
    total_fuel_rot = 0.0

    # Initial alpha
    v_x = env.current_v_r * np.cos(env.current_phi) - env.current_v_theta * np.sin(env.current_phi)
    v_y = env.current_v_r * np.sin(env.current_phi) + env.current_v_theta * np.cos(env.current_phi)
    angle_v = np.arctan2(v_y, v_x)
    alpha = (env.current_psi - angle_v + np.pi) % (2 * np.pi) - np.pi
    alphas.append(alpha)
    psis.append(env.current_psi)

    for i in range(num_sub_steps):
        # Attitude control sub-step
        updated_psi, updated_omega, fuel_use_rot = env.pid_attitude_control(
            env.current_psi,
            env.current_omega,
            env.current_v_r,
            env.current_v_theta,
            env.current_phi,
            target_alpha,
            dt_sub,
            N_sub=1
        )

        # Update state
        env.current_psi = updated_psi
        env.current_omega = updated_omega
        total_fuel_rot += fuel_use_rot

        # Advance orbital angle (mimics the missing integration)
        env.current_phi += (env.current_v_theta / env.current_r) * dt_sub
        env.current_phi %= (2 * np.pi)

        # Increment time and store data
        t = (i + 1) * dt_sub
        times.append(t)

        # Re-compute alpha
        v_x = env.current_v_r * np.cos(env.current_phi) - env.current_v_theta * np.sin(env.current_phi)
        v_y = env.current_v_r * np.sin(env.current_phi) + env.current_v_theta * np.cos(env.current_phi)
        angle_v = np.arctan2(v_y, v_x)
        alpha = (env.current_psi - angle_v + np.pi) % (2 * np.pi) - np.pi

        alphas.append(alpha)
        psis.append(env.current_psi)
        cum_fuel_rot.append(total_fuel_rot)

    if plotting:
        plot_attitude_control_results(times, alphas, psis, cum_fuel_rot, target_alpha, total_fuel_rot)

    print(f"Final alignment error α: {np.rad2deg(alphas[-1]):.2f}°")
    print(f"Total rotational fuel used: {total_fuel_rot:.4f} kg")
    # Print success/failure message:
    if abs((alphas[-1]) - alpha_setpoint) < np.deg2rad(10):
        print("Attitude control successful: Alignment within ±10°.")
    else:
        print("Attitude control failed: Alignment outside ±10°.")
    return alphas, psis, cum_fuel_rot, total_fuel_rot


def plot_attitude_control_results(times, alphas, psis, cum_fuel_rot, target_alpha, total_fuel_rot):

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(times, alphas, label='Alpha (rad)')
    axs[0].axhline(target_alpha, color='r', linestyle='--', label='Set Point')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Alpha (rad)')
    axs[0].set_ylim(-np.pi, np.pi)
    axs[0].set_title('Alignment Error α Over One 600 s Interval')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(times, psis, label='Ψ (rad)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Ψ (rad)')
    axs[1].set_title('Attitude Angle Ψ Over One 600 s Interval')
    axs[1].set_ylim(-np.pi, np.pi)
    axs[1].grid(True)

    axs[2].plot(times, cum_fuel_rot, label='Cumulative Rotational Fuel Use (kg)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Fuel Use (kg)')
    axs[2].set_title('Rotational Thruster Fuel Consumption')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
