from rl_agent import train_agent, save_weights, evaluate_against_random, evaluate_against_greedy
import csv

if __name__ == "__main__":
    TOTAL_GAMES = 4000
    CHECKPOINT = 250
    EVAL_GAMES = 2000

    # Baseline (sem treino)
    base_weights = [0.0, 0.0, 0.0, 0.0]
    baseline_rand = evaluate_against_random(base_weights, num_games=EVAL_GAMES, epsilon=0.0, seed=0)
    baseline_greedy = evaluate_against_greedy(base_weights, num_games=EVAL_GAMES, epsilon=0.0, seed=0)

    print("Baseline vs Random:", baseline_rand)
    print("Baseline vs Greedy:", baseline_greedy)

    stats = []
    trained_weights = None
    current_games = 0

    while current_games < TOTAL_GAMES:
        block = min(CHECKPOINT, TOTAL_GAMES - current_games)

        # Primeiro bloco cria pesos; os próximos continuam via play_game no rl_agent (como já tinhas)
        if current_games == 0:
            trained_weights = train_agent(
                num_games=block,
                epsilon_start=1.0,
                epsilon_end=0.1,
                alpha=0.05
            )
        else:
            from rl_agent import play_game
            for i in range(block):
                t = i / max(1, block - 1)
                epsilon = 1.0 * (1 - t) + 0.1 * t
                play_game(trained_weights, epsilon, 0.05)

        current_games += block

        # Avaliar SEM treino (epsilon=0) com 1000 jogos
        ev_rand = evaluate_against_random(trained_weights, num_games=EVAL_GAMES, epsilon=0.0, seed=0)
        ev_greedy = evaluate_against_greedy(trained_weights, num_games=EVAL_GAMES, epsilon=0.0, seed=0)

        row = {
            "trained_games": current_games,

            "rand_games": ev_rand["games"],
            "rand_wins": ev_rand["wins"],
            "rand_losses": ev_rand["losses"],
            "rand_draws": ev_rand["draws"],
            "rand_win_rate": ev_rand["win_rate"],

            "greedy_games": ev_greedy["games"],
            "greedy_wins": ev_greedy["wins"],
            "greedy_losses": ev_greedy["losses"],
            "greedy_draws": ev_greedy["draws"],
            "greedy_win_rate": ev_greedy["win_rate"],
        }

        stats.append(row)

        print(f"\nApós {current_games} jogos de treino:")
        print("  vs Random :", ev_rand)
        print("  vs Greedy :", ev_greedy)

    # Guardar pesos finais
    save_weights(trained_weights, "weights.json")
    print("\nPesos finais aprendidos:", trained_weights)
    print("Guardado em: weights.json")

    # Guardar CSV completo
    with open("stats.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(stats[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in stats:
            writer.writerow(r)

    print("Guardado em: stats.csv")
