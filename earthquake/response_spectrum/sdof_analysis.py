import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def analyze_sdof_system():
    # System parameters
    m = 1.0  # mass (kg)
    k = 4 * np.pi**2  # stiffness (N/m) chosen to give a natural period of 1 second
    zeta = 0.05  # damping ratio (5%)
    omega_n = np.sqrt(k/m)  # natural frequency (rad/s)
    c = 2 * zeta * np.sqrt(k*m)  # damping coefficient

    # Time parameters
    t = np.linspace(0, 10, 1000)  # 10 seconds with 1000 points

    # Ground motion (simple sine wave)
    freq = 2.0  # Hz
    acc_g = 0.5 * 9.81 * np.sin(2 * np.pi * freq * t)  # 0.5g amplitude

    # Create interpolation function for ground motion
    acc_g_interp = interp1d(t, acc_g, bounds_error=False, fill_value=0.0)

    def sdof_system(t, y):
        # Get ground acceleration at time t using interpolation function
        acc_g_t = acc_g_interp(t)
        
        # y[0] is displacement, y[1] is velocity
        dydt = np.zeros_like(y)
        dydt[0] = y[1]
        dydt[1] = -omega_n**2 * y[0] - 2*zeta*omega_n*y[1] - acc_g_t
        
        return dydt

    # Initial conditions
    y0 = np.array([0.0, 0.0])

    # Solve the system
    sol = solve_ivp(
        sdof_system,
        t_span=[0, 10],
        y0=y0,
        t_eval=t,
        method='RK45',
        rtol=1e-8,
        atol=1e-8
    )

    # Plot results
    plt.style.use('default')  # Use default style instead of seaborn
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Ground acceleration
    ax1.plot(t, acc_g/9.81, 'b-', label='Ground Motion', linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (g)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Ground Motion Input')

    # Relative displacement
    ax2.plot(t, sol.y[0], 'r-', label='Structure Response', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Displacement (m)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('Structural Response')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_sdof_system() 