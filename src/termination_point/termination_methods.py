class Termination_Methods:
    """Example functions that can be used to terminate the the algorithms loop"""

    def fitness_and_generation_based(ga):
        """Fitness based approach to terminate when the goal fitness has been reached"""

        # Need to start the algorithm if the population is None.
        if ga.population == None:
            return True

        # If fitness goal is set, check it.
        if ga.fitness_goal is not None:

            # If minimum fitness goal reached, stop ga.
            if ga.target_fitness_type == 'min' and ga.population.get_chromosome(0).get_fitness() <= ga.fitness_goal:
                return False

            # If maximum fitness goal reached, stop ga.
            elif ga.target_fitness_type == 'max' and ga.population.get_chromosome(0).get_fitness() >= ga.fitness_goal:
                return False

        # If generation goal is set, check it.
        if ga.generation_goal is not None and ga.current_generation >= ga.generation_goal:
            return False

        # If tolerance is set, check it.
        if ga.tolerance_goal is not None:

            best_fitness = ga.population.get_chromosome(0).get_fitness()
            threshhold_fitness = ga.population.get_chromosome(int(0.1*ga.population.size())).get_fitness()
            tol = ga.tolerance_goal * (1 + abs(best_fitness))

            # Terminate if 10% of the population has converged
            if abs(best_fitness - threshhold_fitness) < tol:
                return False

        return True
