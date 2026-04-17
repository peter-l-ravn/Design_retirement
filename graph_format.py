import matplotlib.pyplot as plt

def plot_wages(wages_sim_exp, wages_ext):
    x = range(30, 30 + len(wages_sim_exp))

    plt.figure(figsize=(8, 4))
    plt.plot(x, wages_sim_exp, color="blue", linestyle="--", linewidth=2, label="Simulated Wage")
    plt.plot(x, wages_ext, color="red", linestyle="-", linewidth=2, label="Actual Wage")
    plt.xlabel("Age")
    plt.ylabel("Wage")
    plt.legend()
    plt.tight_layout()
    plt.show()