import marimo

__generated_with = "0.10.2"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    # Grid setup
    X, Y = np.meshgrid(np.linspace(-2,2, 200), np.linspace(-2,2, 200))
    q, d = 1, 0.5  # Charge magnitude and half-distance
    charges = [(-d, 0, 1.0*q), (d, 0, -1.0*q)]  # (x, y, charge)

    # Electric field function
    def efield(X, Y, charges):
        Ex, Ey = np.zeros_like(X), np.zeros_like(Y)
        for x_c, y_c, q in charges:
            r = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
            r3 = np.maximum(r**3, 1e-12)  # Avoid division by zero
            Ex += q * (X - x_c) / r3
            Ey += q * (Y - y_c) / r3
        return Ex, Ey

    # Start points for field lines (required for E-Field line density)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    start_radius = 0.3
    start_points = np.vstack([
        [x_c + start_radius * np.cos(angles), y_c + start_radius * np.sin(angles)] 
        for x_c, y_c, _ in charges
    ]).T.reshape(-1, 2)

    # Calculate field and plot
    Ex, Ey = efield(X, Y, charges)

    # Create the streamplot
    plt.figure(figsize=(5,5))
    plt.streamplot(X, Y, Ex, Ey, linewidth=1, start_points=start_points, 
                   integration_direction='both',density=5, zorder=1)
    # Mark the charges
    plt.scatter([-d, d], [0, 0], color=['red', 'blue'], s=100, zorder=2)  
    plt.text(-d, 0.1, '+', color='red', fontsize=18, ha='center', zorder=2)
    plt.text(d, 0.1, '-', color='blue', fontsize=18, ha='center', zorder=2)
    plt.title("Electric Field - Dipole")
    plt.axis('equal')
    plt.show()
    return (
        Ex,
        Ey,
        X,
        Y,
        angles,
        charges,
        d,
        efield,
        np,
        plt,
        q,
        start_points,
        start_radius,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
