"""
Create an iris flow
"""
import numpy as np
from prefect import flow

from config import Location, ModelParams, ProcessConfig
from optimize.optimizer import SimulatedAnnealingOptimizer
from process import process
from run_notebook import run_notebook
from train_model import train


@flow
def iris_flow(
    location: Location = Location(),
    process_config: ProcessConfig = ProcessConfig(),
    model_params: ModelParams = ModelParams(),
):
    """Flow to run the process, train, and run_notebook flows

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    process_config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    model_params : ModelParams, optional
        Configurations for training models, by default ModelParams()
    """
    process(location, process_config)
    train(location, model_params)
    run_notebook(location)


if __name__ == "__main__":
    # Define a simple function to optimize
    def simple_func(x: np.ndarray) -> float:
        """Simple function to optimize
            Parameters
            ----------
        x : numpy array
            coefficients of each variable
        """
        return x[0] ** 2 - np.log(x[1]) + x[1] ** x[2]

    # Define initial parameters
    initial_params = np.array([1, 2, 2])

    # Create an instance of the SimulatedAnnealingOptimizer class
    sa = SimulatedAnnealingOptimizer(
        temperature=20,
        cooling_rate=0.95,
        num_iterations=1000,
        initial_params=initial_params,
        loss_func=simple_func,
    )

    # Optimize the function using simulated annealing
    sa.optimize()

    # Print the best parameters and loss found during optimization
    print("Best parameters: ", sa.get_best_params())
    print("Best loss: ", sa.get_best_loss())
