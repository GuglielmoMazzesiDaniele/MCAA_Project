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
            'std_errors': std_error_array,
            'successes': int
        }
    }
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(energies_dict)))

    for (name, data), color in zip(energies_dict.items(), colors):
        energy = np.array(data['energy'])
        std_errors = np.array(data.get('std_errors', np.zeros_like(energy)))
        successes = data['successes']
        
        iterations = np.arange(0, len(energy), 100)
        energy_plot = energy[::100]
        std_errors_plot = std_errors[::100]

        label = f"{name} (success = {successes})"
        ax.plot(iterations, energy_plot, label=label, color=color, linewidth=2)
        
        ax.fill_between(iterations, 
                        energy_plot - std_errors_plot, 
                        energy_plot + std_errors_plot,
                        alpha=0.2, color=color)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Energy")
    ax.set_title(f"Average Energy Evolution for Different Schedulers ({n_runs} runs each)")
    ax.legend()

    plt.subplots_adjust(left=0.28)
    plt.grid(True)
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
    

def plot_vary_n(all_minimal_energies, number_of_successes, args, name_proposal_move,
                n_min, n_max, n_runs, filename="vary_n_comparison.png"):

    Ns = np.arange(n_min, n_max + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(Ns, all_minimal_energies, marker='o', label="Minimal Energies")
    ax.set_xlabel("N")
    ax.set_ylabel("Minimal Energy")
    ax.set_title(f"Minimal Energy per N ({n_runs} runs each)")
    ax.legend()

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

    ax.set_ylim(ymin - 0.12*(ymax - ymin), ymax)
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
