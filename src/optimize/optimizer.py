from typing import Callable, List

import numpy as np
from pulp import LpMinimize, LpProblem, LpVariable, lpSum


class SimulatedAnnealingOptimizer:
    temperature: float
    cooling_rate: float
    num_iterations: int
    initial_params: np.ndarray
    loss_func: Callable[[np.ndarray], float]
    best_params: np.ndarray
    best_loss: float

    """
    Simulated Annealing class for optimizing an input function.
    This optimizer refers to the metalurgic thermodynamic process.

    Simulated Annealing techique is better suited for problems
    where finding an approximate global optimum is more important than
    finding a precise local optimum in a fixed amount of time, simulated annealing
    may be preferable to exact algorithms such as gradient descent or branch and bound.

    Parameters:
    -----------
    temperature : float
        The starting temperature for the simulated annealing algorithm.
    cooling_rate : float
        The cooling rate for the simulated annealing algorithm.
        This determines how quickly the temperature is reduced over time.
    num_iterations : int
        The number of iterations to run the simulated annealing algorithm for.
    initial_params : numpy array
        The initial parameters for the function to be optimized.
    loss_func : function
        The function to be optimized.

    Attributes:
    -----------
    temperature : float
        The current temperature for the simulated annealing algorithm.
    cooling_rate : float
        The cooling rate for the simulated annealing algorithm.
    num_iterations : int
        The number of iterations to run the simulated annealing algorithm for.
    initial_params : numpy array
        The initial parameters for the function to be optimized.
    loss_func : function
        The function to be optimized.
    best_params : numpy array
        The best parameters found during optimization.
    best_loss : float
        The best loss found during optimization.

    Methods:
    --------
    optimize()
        Run the simulated annealing algorithm to optimize the function.
        This could take some time... Let's have a coffee/tea!
    get_best_params()
        Return the best parameters found during optimization.
    get_best_loss()
        Return the best loss found during optimization.
    """

    def __init__(
        self,
        temperature: float,
        cooling_rate: float,
        num_iterations: int,
        initial_params: np.ndarray,
        loss_func: Callable[[np.ndarray], float],
    ) -> None:
        """
        Init method for SimulatedAnnealingOptimizer class
        """
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.initial_params = initial_params
        self.loss_func = loss_func
        self.best_params = None
        self.best_loss = np.inf

    def optimize(self) -> None:
        """
        Run the simulated annealing algorithm to optimize the function.
        """
        params = self.initial_params
        for i in range(self.num_iterations):
            neighbor_params = self._get_neighbor_params(params)
            loss = self.loss_func(params)
            neighbor_loss = self.loss_func(neighbor_params)
            acceptance_prob = self._get_acceptance_prob(
                loss, neighbor_loss, self.temperature
            )

            if acceptance_prob > np.random.uniform(0, 1):
                params = neighbor_params

            if loss < self.best_loss:
                self.best_params = params
                self.best_loss = loss

            self.temperature *= self.cooling_rate

    def get_best_params(self) -> np.ndarray:
        """
        Return the best parameters found during optimization.

        Returns:
        --------
        numpy array
            The best parameters found during optimization.
        """
        return self.best_params

    def get_best_loss(self) -> float:
        """
        Return the best loss found during optimization.

        Returns:
        --------
        float
            The best loss found during optimization.
        """
        return self.best_loss

    def _get_neighbor_params(self, params) -> np.ndarray:
        """
        Gets the parameters of a neighbor of the given parameters.

        Args:
            params (np.ndarray): The current parameters.

        Returns:
            np.ndarray: The neighbor parameters.
        """
        return params + np.random.normal(0, 1, params.shape)

    def _get_acceptance_prob(self, loss, neighbor_loss, temperature) -> float:
        """
        Calculates the acceptance probability of a new set of parameters based on the
        change in loss and temperature.

        Args:
            new_loss (float): The loss of the new parameters.
            old_loss (float): The loss of the current parameters.
            temperature (float): The current temperature.

        Returns:
            float: The acceptance probability.
        """

        # Define a simple acceptance probability function using Boltzmann distribution
        if neighbor_loss < loss:
            return 1
        else:
            return np.exp(-(neighbor_loss - loss) / temperature)


class LinearProgrammingOptimizer:
    num_vars: int
    constraints: List[List[float]]
    coefficients: List[float]
    variable_names: List[str]
    solution: List[float]
    objective_value: float

    def __init__(
        self,
        num_vars: int,
        constraints: List[List[float]],
        coefficients: List[float],
        variable_names: List[str],
    ) -> None:
        """
        Initializes the LinearProgrammingOptimizer class.

        Args:
            num_vars (int): The number of decision variables.
            constraints (List[List[float]]): The matrix of coefficients for the constraints.
            coefficients (List[float]): The coefficients of the objective function.
            variable_names (List[str]): The names of the decision variables.
        """
        self.num_vars = num_vars
        self.constraints = constraints
        self.coefficients = coefficients
        self.variable_names = variable_names

    def optimize(self) -> None:
        """
        Runs the optimization algorithm.
        """
        # Create the problem
        problem = LpProblem("LP_Problem", LpMinimize)

        # Define decision variables
        vars = LpVariable.dicts("x", self.variable_names, lowBound=0)

        # Define the objective function
        problem += lpSum(
            [
                self.coefficients[i] * vars[self.variable_names[i]]
                for i in range(self.num_vars)
            ]
        )

        # Define the constraints
        for i in range(len(self.constraints)):
            problem += (
                lpSum(
                    [
                        self.constraints[i][j] * vars[self.variable_names[j]]
                        for j in range(self.num_vars)
                    ]
                )
                <= self.constraints[i][-1]
            )

        # Solve the problem
        problem.solve()

        # Save the solution
        self.solution = [
            vars[self.variable_names[i]].value() for i in range(self.num_vars)
        ]
        self.objective_value = problem.objective.value()

    def get_solution(self) -> List[float]:
        """
        Returns the optimal solution found during optimization.

        Returns:
            List[float]: The optimal solution.
        """
        return self.solution

    def get_objective_value(self) -> float:
        """
        Returns the objective value of the optimal solution found during optimization.

        Returns:
            float: The objective value.
        """
        return self.objective_value
