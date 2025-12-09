import matplotlib.pyplot as plt
import numpy as np

def plot_energy_evolution(energies, args, name_proposal_move, filename):
    """Plot the energy (every 100 element) evolution during the pipeline run."""
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(energies[::100])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Evolution")

    plt.subplots_adjust(left=0.25)

    param_text = (
        f"N = {args.N}\n"
        f"K = {args.K}\n"
        f"beta = {args.beta}\n"
        f"max_iters = {args.max_iters}\n"
        f"reheating = {args.reheating}\n"
        f"patience = {args.patience}\n"
        f"move = {name_proposal_move}"
    )

    fig.text(
        0.02, 0.5, param_text,
        va="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85)
    )

    plt.savefig(filename, dpi=300)
    plt.close()
    
def plot_3d_queens(config, args, filename):
    """Draw a 3D NxNxN chessboard and the N^2 queens as red spheres/dots."""

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates
    xs = config[:, 0]
    ys = config[:, 1]
    zs = config[:, 2]

    # --- Draw queens ---
    ax.scatter(xs, ys, zs, c='red', s=60, depthshade=True, label="Queens")

    # --- Draw board edges ---
    r = range(args.N)
    for s in [0, args.N - 1]:
        # Faces perpendicular to x
        xx, yy = np.meshgrid([s], r)
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.05)
        ax.plot_surface(xx, yy, np.ones_like(xx)*(args.N-1), alpha=0.05)

        # Faces perpendicular to y
        xx, yy = np.meshgrid(r, [s])
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.05)
        ax.plot_surface(xx, yy, np.ones_like(xx)*(args.N-1), alpha=0.05)

        # Faces perpendicular to z
        xx, yy = np.meshgrid(r, r)
        ax.plot_surface(xx, yy, np.ones_like(xx)*s, alpha=0.03)

    # --- Style ---
    ax.set_xlim(0, args.N-1)
    ax.set_ylim(0, args.N-1)
    ax.set_zlim(0, args.N-1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(filename)
    plt.savefig(filename, dpi=300)
    plt.tight_layout()
    plt.show()