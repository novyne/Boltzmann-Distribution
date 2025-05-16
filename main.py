import cProfile

from numpy import random

from matplotlib import pyplot

def get_shift(shift_map: dict[float, int], x: float) -> int | None:
    """
    Return the shift value in the given shift map for the given x value.
    The shift value is the one whose key is the smallest value in the shift map
    that is greater than or equal to x. If no such value exists, return None.

    Args:
        shift_map (dict[float, int]): A dictionary mapping x values to shift values.
        x (float): The x value to get the shift value for.

    Returns:
        int | None: The shift value for the given x value, or None if no such value exists.
    """

    for key in shift_map:
        if x < key:
            return shift_map[key]
    return None

def midpoint(trials: int, shift_map: dict[float, int]) -> int:
    """
    Determine the midpoint of the Boltzmann distribution.

    Args:
        trials (int): The number of trials.
        shift_map (dict[float, int]): A dictionary mapping x values to shift values.

    Returns:
        float: The midpoint of the Boltzmann distribution.
    """

    highest_shift = max(shift_map.values())
    return round(trials / 2 * highest_shift)

def B(unit_number: int=100, trials: int=100, shift_map: dict[float, int]={0.5 : -1, 1 : 1}) -> dict[int,int]:
    """
    Simulate the Boltzmann distribution and return the population of units after
    a given number of trials. The Boltzmann distribution is a probability
    distribution that assigns a probability of 1 to the midpoint of the shift
    map and a probability of 0 to the edges of the shift map. The probability
    of a unit being shifted is proportional to the value of the shift map at
    the unit's current position. The unit is shifted by the value of the shift
    map at the unit's current position.

    Args:
        unit_number (int): The number of units to simulate. Defaults to 100.
        trials (int): The number of trials to run. Defaults to 100.
        shift_map (dict[float, int]): A dictionary mapping x values to shift
            values. Defaults to {0.5 : -1, 1 : 1}.

    Returns:
        dict[int,int]: A dictionary mapping the population of each unit to the
            number of units with that population.
    """
    
    mid = midpoint(trials, shift_map)
    units = [mid] * unit_number

    for _ in range(trials):

        randoms = random.random(unit_number)

        for i in range(unit_number):

            x = randoms[i]
            shift = get_shift(shift_map, x)
            if shift is None:
                raise ValueError(f"Could not find shift for x={x}")
            
            # Shift the unit
            units[i] += shift

    # Convert list of units into dictionary of population
    results = {}
    for u in sorted(units):
        results[u] = results.get(u, 0) + 1

    return results


def save_csv(results: dict[int,int], filename: str="boltzmann.csv"):
    with open(filename, "w") as f:
        for k, v in results.items():
            f.write(f"{k},{v}\n")

def plot(results: dict[int,int]):
    """
    Plot a Boltzmann distribution.

    Normalise the given results, then smooth by averaging with the given number
    of neighbours either side of each point. Finally, plot the Boltzmann
    distribution and save it to a file called "boltzmann.png".

    Args:
        results (dict[int,int]): A dictionary mapping x values to population
            values.
    """

    # Normalise
    total = sum(results.values())
    results = {k : v / total for k, v in results.items()}

    # Smooth with a neighbouring average
    neighbours_either_side = 3
    for k, _ in results.items():
        average = 0
        
        neighbours = [results.get(k + i) for i in range(-neighbours_either_side, neighbours_either_side + 1)]
        neighbours = [n for n in neighbours if n is not None]
        average = sum(neighbours) / len(neighbours)
        results[k] = average

    # Plot
    pyplot.plot(results.keys(), results.values())
    pyplot.savefig("boltzmann.png")


def main():

    shift_map = {1 / 6 : -3, 2 / 6 : -2, 3 / 6 : -1, 4 / 6 : 1, 5 / 6 : 2, 6 / 6 : 3}
    results = B(unit_number=10000, trials=50, shift_map=shift_map)

    print("Complete. Plotting...")

    # plot(results)
    # save_csv(results)

if __name__ == "__main__":
    # main()
    cProfile.run("main()", sort="cumulative")