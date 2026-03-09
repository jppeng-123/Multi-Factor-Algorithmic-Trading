Genetic Algorithm Alpha Mining

This project explores the use of genetic programming to automatically discover nonlinear alpha signals from equity market data.

Instead of manually designing factors, the idea is to let an evolutionary search process generate candidate formulas, evaluate their predictive power, and iteratively improve them across generations.

The goal is not to find a single “perfect factor,” but to build a pipeline that systematically searches the space of possible signals and identifies statistically meaningful candidates.

Motivation

Many equity factors used in systematic strategies originate from relatively simple transformations of price, volume, or return series.

However, the search space of possible combinations quickly becomes enormous. A genetic algorithm provides a practical way to explore this space by:

generating candidate formulas

evaluating their predictive power

selecting stronger signals

recombining them into new expressions

Over time, the population of formulas evolves toward signals with stronger out-of-sample performance.

Approach

The system represents each alpha candidate as a tree-structured expression composed of price/volume features and rolling operators.

Examples of operators include:

rolling mean

rolling standard deviation

rank transformations

basic arithmetic operations

Each expression acts as a potential alpha signal.

The search process evolves these expressions across generations using:

mutation (random structural changes)

crossover (combining parts of two formulas)

tournament selection

elite retention

To avoid overfitting, the search is constrained with:

maximum expression depth

limited rolling windows (5 / 10 / 20 / 60 days)

complexity penalties in the fitness score

Walk-Forward Evaluation

Signals are evaluated using a walk-forward validation framework.

Instead of testing on a single train/test split, the model repeatedly retrains and evaluates signals across rolling time windows.

Each evaluation step uses:

training window → validation window

with purge gaps to prevent look-ahead bias and leakage between samples.

This approach better reflects how signals behave in real trading environments.

Signal Evaluation

Each candidate alpha is evaluated using several cross-sectional metrics:

Information Coefficient (IC)

IC Information Ratio (ICIR)

Newey-West adjusted t-statistics

These metrics measure whether a signal has consistent predictive power across stocks in each cross-section.

Before evaluation, signals are:

cross-sectionally normalized

industry neutralized

to isolate true predictive information rather than sector exposure.

Search Scale

In the current experiments:

384 candidate formulas are evaluated per generation

180 generations are evolved

roughly 69,000 formulas are explored

The evolutionary process gradually improves signal quality while maintaining diversity in the population.

Results

The search process produced several alpha candidates exhibiting:

statistically significant predictive relationships

Newey-West t-statistics above 2

stable cross-sectional IC behavior across evaluation windows

Rather than selecting a single signal, the pipeline generates a library of candidate alphas that can be further tested, combined, or incorporated into multi-factor models.

Technologies Used

Python ecosystem:

NumPy

pandas

scikit-learn

custom genetic programming implementation

The system is designed to be modular so the evaluation framework and evolutionary engine can be reused for other factor mining experiments.

Future Improvements

Possible extensions include:

expanding the operator set

incorporating alternative data features

integrating transaction cost modeling

combining evolved signals into ensemble factor models
