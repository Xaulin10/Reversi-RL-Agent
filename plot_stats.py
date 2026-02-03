import csv
import matplotlib.pyplot as plt

def main():
    xs = []
    rand = []
    greedy = []

    with open("stats.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(int(row["trained_games"]))
            rand.append(float(row["rand_win_rate"]))
            greedy.append(float(row["greedy_win_rate"]))

    plt.figure()
    plt.plot(xs, rand, marker="o", label="vs Random")
    plt.plot(xs, greedy, marker="o", label="vs Greedy")
    plt.title("Curvas de Aprendizagem do Agente RL (Reversi)")
    plt.xlabel("Jogos de treino")
    plt.ylabel("Win rate (avaliação)")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    plt.savefig("learning_curve.png", dpi=200, bbox_inches="tight")
    print("Gráfico gerado: learning_curve.png")

if __name__ == "__main__":
    main()
