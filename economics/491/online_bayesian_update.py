from sympy import symbols, integrate, binomial, simplify

EPS = 0.01

class BayesianUpdater:
    def __init__(self, distribution, symbol):
        assert abs(1 - integrate(distribution, (symbol, 0, 1))) < EPS

        self.distribution = distribution
        self.symbol = symbol

    def apply_realization(self, trials, successes):
        likelihood = binomial(trials, successes) * (self.symbol ** successes) * ((1 - self.symbol) ** (trials -successes))

        kernel = likelihood * self.distribution
        normalization_constant = integrate(kernel, (self.symbol, 0, 1))

        self.distribution = simplify( kernel / normalization_constant )

    def print(self):
        print(f'{self.distribution}')


if __name__ == '__main__':
    x = symbols('x')

    example = BayesianUpdater(3 * (x ** 2), x)
    example.print()

    example.apply_realization(10, 0)
    example.print()
