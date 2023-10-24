# Description: Utilities used to track execution time.
from time import perf_counter


def get_and_reset_time(initial_timer: float) -> float:
    """
    Gets the time elapsed since <initial_timer> was set and resets it.

    Args:
        initial_timer (float): Initial time.

    Returns:
        float: Time elapsed since <initial_timer> was set.
    """
    current_time = perf_counter()
    elapsed_time = current_time - initial_timer
    initial_timer = current_time
    return elapsed_time, initial_timer
