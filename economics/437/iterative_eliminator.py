from typing import Optional

Player = int
Strategy = str
Game = dict[tuple[Strategy, Strategy], tuple[float, float]]

def find_first_dominated_strategy(game: Game) -> Optional[tuple[Player, Strategy]]:
    p1_strategies: set[Strategy] = {
        v for v, _ in game.keys()
    }

    p2_strategies: set[Strategy] = {
        v for _, v in game.keys()
    }

    optimal_p1_payoff: dict[Strategy, float] = dict()
    optimal_p2_payoff: dict[Strategy, float] = dict()

    for strategies, payoffs in game.items():
        if strategies[1] not in optimal_p1_payoff.keys():
            optimal_p1_payoff[strategies[1]] = payoffs[0]
        else:
            optimal_p1_payoff[strategies[1]] = max(
                optimal_p1_payoff[strategies[1]],
                payoffs[0]
            )       

        if strategies[0] not in optimal_p2_payoff.keys():
            optimal_p2_payoff[strategies[0]] = payoffs[1]
        else:
            optimal_p2_payoff[strategies[0]] = max(
                optimal_p2_payoff[strategies[0]],
                payoffs[1]
            )
    
    for p1_strat in p1_strategies:
        dominated = True
        for p2_strat in p2_strategies:
            if game[(p1_strat, p2_strat)][0] == optimal_p1_payoff[p2_strat]:
                dominated = False

        if dominated:
            return (0, p1_strat)

    for p2_strat in p2_strategies:
        dominated = True
        for p1_strat in p1_strategies:
            if game[((p1_strat, p2_strat))][1] == optimal_p2_payoff[p1_strat]:
                dominated = False

        if dominated:
            return (1, p2_strat)

    return None


def eliminate_options(game: Game, finder):
    while dominated_strategy := finder(game):        
        game = {
            u : v for (u, v) in game.items() if u[dominated_strategy[0]] != dominated_strategy[1]
        }
    return game


if __name__ == '__main__':
    Simple_P1 = Game({
        ("Stag", "Stag") : (4, 4),
        ("Stag", "Hare") : (4, 3),
        ("Hare", "Stag") : (1, 0),
        ("Hare", "Hare") : (1, 2)
    })

    # why does python make it so hard to tell when side effects happen...
    print(eliminate_options(Simple_P1, find_first_dominated_strategy))
    