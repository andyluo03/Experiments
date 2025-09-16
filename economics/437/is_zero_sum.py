"""
In game theory, zero-sum games have an interesting property: the mini-max strategy will produce a nash equlibrium. 

In single-turn games, this means we should randomize our strategies such that it is optimal for the opponent to 
pick all strategies as equal probabilitiy. 

However, not all zero-sum games *look* like zero-sum games. For any game in the set of zero-sum games,
applying the following transformations:

-- multiplication (by a positive number)
-- addition 

is also in the set of zero-sum games. Additionally, note, these operations are invertible. 

Lemma: any legal transformation can be represented in the form f(x) = ax + b.
    Let s(n) represent the set of all transformations consisting of n operations.

    Base case: s(0) --> a = 0, b = 0
    IH: all s(n - 1) can be represented by ax + b
    IS: 
        if operation[n] is '* n':
            operation[1..n-1] in s(n-1), representable by f(x) = a'x + b',
            so we can represent as f(x) = (a' * n) x + (b' * n)
        if operation[n] is '+ n':
            operation[1..n-1] in s(n-1), representable by f(x) = a'x + b',
            so we can represent as f(x) = a'x + (b' + n)

Thus, using linear algebra, we can produce constraints and solve for valid transformations!
"""

from typing import Optional
from sympy import Matrix
from pprint import pprint

Game = dict[tuple, tuple]

def find_zero_sum_transformation(game: Game) -> Optional[Matrix]:
    # 1. Construct matrix of constraints
    rows = []
    for payoff in game.values():
        rows.append([payoff[0] + payoff[1], 2])

    payoff_constraint_matrix = Matrix(rows)

    # 2. Return nullspace.
    projection_basis = payoff_constraint_matrix.nullspace()

    if len(projection_basis) == 0:
        return None
    else:
        return projection_basis

def apply_transformation(game: Game, transformation: Matrix) -> Game:
    if transformation[0][0] < 0:
        transformation[0][0] *= -1
        transformation[0][1] *= -1

    return {
        k : (
            v[0] * transformation[0][0] + transformation[0][1],
            v[1] * transformation[0][0] + transformation[0][1]
        ) for k, v in game.items()
    }

if __name__ == '__main__':
    RPS_scaled = Game({
        ("Rock", "Rock") : (1, 1),
        ("Rock", "Paper") : (0, 2),
        ("Rock", "Scissors") : (2, 0),
        ("Paper", "Rock") : (2, 0),
        ("Paper", "Paper") : (1, 1),
        ("Paper", "Scissors") : (0, 2),
        ("Scissors", "Rock") : (0, 2),
        ("Scissors", "Paper") : (2, 0),
        ("Scissors", "Scissors") : (1, 1)   
    })

    RPS_zero_sum_projection_basis = find_zero_sum_transformation(RPS_scaled)
    if RPS_zero_sum_projection_basis == None:
        print('RPS is not zero-sum')
    else:
        print(f'RPS is zero-sum with transformation: f(x) = {RPS_zero_sum_projection_basis[0][0]}x + {RPS_zero_sum_projection_basis[0][1]}!')
        pprint(apply_transformation(RPS_scaled, RPS_zero_sum_projection_basis))

    Cooperative_Hunting = Game({
        ("Stag", "Stag") : (4, 4),
        ("Stag", "Hare") : (0, 3),
        ("Hare", "Stag") : (3, 0),
        ("Hare", "Hare") : (2, 2)
    })
    Cooperative_Hunting_zero_sum_projection_basis = find_zero_sum_transformation(Cooperative_Hunting)
    assert Cooperative_Hunting_zero_sum_projection_basis == None
