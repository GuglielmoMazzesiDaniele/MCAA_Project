import matplotlib.pyplot as plt
import numpy as np

def plot_energy_evolution(energies, successes, args, name_proposal_move, filename):
    """Plot the energy (every 100 element) evolution during the pipeline run."""
    
    fig, ax = plt.subplots(figsize=(10, 6))

    label = f"(success = {successes})"
    ax.plot(energies[::100], label=label)
    ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Evolution")

    plt.subplots_adjust(left=0.25)

    patience_display = args.patience if args.reheating else 0

    param_text = (
        f"N = {args.N}\n"
        f"K = {args.K}\n"
        f"beta = {args.beta}\n"
        f"max_iters = {args.max_iters}\n"
        f"reheating = {args.reheating}\n"
        f"patience = {patience_display}\n"
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
    
def plot_all_schedulers(energies_dict, args, name_proposal_move, n_runs, filename):
    """
    Plot the energy evolution for all schedulers on a single plot.
    
    energies_dict: {
        scheduler_name : {
            'energy': energy_array,
            'successes': int
        }
    }
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each scheduler
    for name, data in energies_dict.items():
        energy = data['energy']
        successes = data['successes']

        label = f"{name} (success = {successes})"
        ax.plot(energy[::100], label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Energy")
    ax.set_title(f"Average Energy Evolution for Different Schedulers ({n_runs} runs each)")
    ax.legend()

    # Add parameter box on the left
    plt.subplots_adjust(left=0.28)

    patience_display = args.patience if args.reheating else 0

    param_text = (
        f"N = {args.N}\n"
        f"K = {args.K}\n"
        f"beta = {args.beta}\n"
        f"max_iters = {args.max_iters}\n"
        f"reheating = {args.reheating}\n"
        f"patience = {patience_display}\n"
        f"move = {name_proposal_move}"
    )

    fig.text(
        0.02, 0.5, param_text,
        va="center", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85)
    )

    plt.savefig(filename, dpi=300)
    plt.close()
    
import matplotlib.pyplot as plt
import numpy as np

def plot_vary_n(all_minimal_energies, number_of_successes, args, name_proposal_move,
                n_min, n_max, n_runs, filename="vary_n_comparison.png"):

    Ns = np.arange(n_min, n_max + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Plot minimal energies ---
    ax.plot(Ns, all_minimal_energies, marker='o', label="Minimal Energies")
    ax.set_xlabel("N")
    ax.set_ylabel("Minimal Energy")
    ax.set_title(f"Minimal Energy per N ({n_runs} runs each)")
    ax.legend()

    # --- Add success info below each N ---
    ymin, ymax = ax.get_ylim()
    y_text_level = ymin - 0.05*(ymax - ymin)

    for x, s in zip(Ns, number_of_successes):
        color = "green" if s > 0 else "red"
        ax.text(
            x, y_text_level,
            f"{s}/{n_runs}",
            ha='center',
            va='top',
            fontsize=9,
            color=color,
            bbox=dict(facecolor='white', edgecolor=color, boxstyle="round,pad=0.2")
        )

    # Extend limits to show text box
    ax.set_ylim(ymin - 0.12*(ymax - ymin), ymax)

    # --- Parameter box on the left ---
    plt.subplots_adjust(left=0.28)

    patience_display = args.patience if args.reheating else 0

    param_text = (
        f"n_min = {n_min}\n"
        f"n_max = {n_max}\n"
        f"runs = {n_runs}\n"
        f"K = {args.K}\n"
        f"beta = {args.beta}\n"
        f"max_iters = {args.max_iters}\n"
        f"reheating = {args.reheating}\n"
        f"patience = {patience_display}\n"
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