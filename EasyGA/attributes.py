from __future__ import annotations
from inspect import getmro, signature
from typing import Any, Callable, Dict, Iterable, Iterator, Optional
from math import sqrt, ceil
from dataclasses import dataclass, field, _MISSING_TYPE
from types import MethodType
import random

import sqlite3
import matplotlib.pyplot as plt

from structure import Population
from structure import Chromosome
from structure import Gene

from examples import Fitness
from termination import Termination
from parent import Parent
from survivor import Survivor
from crossover import Crossover
from mutation import Mutation
from database import SQLDatabase, MatplotlibGraph, SQLDatabase as Database, MatplotlibGraph as Graph

#========================================#
# Default methods not defined elsewhere. #
#========================================#

def rand_1_to_10(self: Attributes) -> int:
    """
    Default gene_impl, returning a random integer from 1 to 10.

    Returns
    -------
    rand : int
        A random integer between 1 and 10, inclusive.
    """
    return random.randint(1, 10)

def use_genes(self: Attributes) -> Iterator[Any]:
    """
    Default chromosome_impl, generates a chromosome using the gene_impl and chromosome length.

    Attributes
    ----------
    gene_impl() -> Any
        A gene implementation.
    chromosome_length : int
        The length of a chromosome.

    Returns
    -------
    chromosome : Iterator[Any]
        Generates the genes for a chromosome.
    """
    for _ in range(self.chromosome_length):
        yield self.gene_impl()

def use_chromosomes(self: Attributes) -> Iterator[Iterable[Any]]:
    """
    Default population_impl, generates a population using the chromosome_impl and population size.

    Attributes
    ----------
    chromosome_impl() -> Any
        A chromosome implementation.
    population_size : int
        The size of the population.

    Returns
    -------
    population : Iterator[Iterable[Any]]
        Generates the chromosomes for a population.
    """
    for _ in range(self.population_size):
        yield self.chromosome_impl()

def dist_fitness(self: Attributes, chromosome_1: Chromosome, chromosome_2: Chromosome) -> float:
    """
    Measures the distance between two chromosomes based on their fitnesses.

    Parameters
    ----------
    chromosome_1, chromosome_2 : Chromosome
        Chromosomes being compared.

    Returns
    -------
    dist : float
        The distance between the two chromosomes.
    """
    return sqrt(abs(chromosome_1.fitness - chromosome_2.fitness))

def simple_linear(self: Attributes, weight: float) -> float:
    """
    Returns a random value between 0 and 1, with increased probability
    closer towards the side with weight.

    Parameters
    ----------
    weight : float
        A float between 0 and 1 which determines the output distribution.

    Returns
    -------
    rand : float
        A random value between 0 and 1.
    """
    rand = random.random()
    if rand < weight:
        return rand * (1-weight) / weight
    else:
        return 1 - (1-rand) * weight / (1-weight)


@dataclass
class AttributesData:
    """
    Attributes class which stores all attributes in a dataclass.
    This includes type-hints/annotations and default values, except for methods.

    Additionally gains dataclass features, including an __init__ and __repr__ to avoid boilerplate code.

    Developer Notes:

        See the Attributes class for default methods.

        Override this class to set default attributes. See help(Attributes) for more information.

        If you must override the __post_init__, don't forget to use super().__post_init__().
    """

    run: int = 0

    chromosome_length: int = 10
    population_size: int = 10
    population: Optional[Population] = None

    target_fitness_type: str = 'max'
    update_fitness: bool = False

    parent_ratio: float = 0.1
    selection_probability: float = 0.5
    tournament_size_ratio: float = 0.1

    current_generation: int = 0
    generation_goal: int = 100
    fitness_goal: Optional[float] = None
    tolerance_goal: Optional[float] = None
    percent_converged: float = 0.5

    chromosome_mutation_rate: float = 0.15
    gene_mutation_rate: float = 0.05

    adapt_rate: float = 0.05
    adapt_probability_rate: float = 0.05
    adapt_population_flag: bool = True

    max_selection_probability: float = 0.75
    min_selection_probability: float = 0.25
    max_chromosome_mutation_rate: float = None
    min_chromosome_mutation_rate: float = None
    max_gene_mutation_rate: float = 0.15
    min_gene_mutation_rate: float = 0.01

    #=================================#
    # Default methods are implemented #
    # in the Attributes descriptors:  #
    #=================================#

    fitness_function_impl: Callable[["Attributes", Chromosome], float] = None
    make_gene: Callable[[Any], Gene] = None
    make_chromosome: Callable[[Iterable[Any]], Chromosome] = None
    make_population: Callable[[Iterable[Iterable[Any]]], Population] = None

    gene_impl: Callable[[], Any] = None
    chromosome_impl: Callable[[], Iterable[Any]] = None
    population_impl: Callable[[], Iterable[Iterable[Any]]] = None

    weighted_random: Callable[[float], float] = None
    dist: Callable[["Attributes", Chromosome, Chromosome], None] = None

    parent_selection_impl: Callable[["Attributes"], None] = None
    crossover_individual_impl: Callable[["Attributes"], None] = None
    crossover_population_impl: Callable[["Attributes", Chromosome, Chromosome], None] = None
    survivor_selection_impl: Callable[["Attributes"], None] = None
    mutation_individual_impl: Callable[["Attributes", Chromosome], None] = None
    mutation_population_impl: Callable[["Attributes"], None] = None
    termination_impl: Callable[["Attributes"], bool] = None

    database: Database = field(default_factory=SQLDatabase)
    database_name: str = "database.db"
    save_data: bool = True
    sql_create_data_structure: str = """
        CREATE TABLE IF NOT EXISTS data (
            id INTEGER PRIMARY KEY,
            config_id INTEGER DEFAULT NULL,
            generation INTEGER NOT NULL,
            fitness REAL,
            chromosome TEXT
        );
    """

    graph: Callable[[Database], Graph] = MatplotlibGraph

    def __post_init__(self: AttributesData) -> None:
        """
        Undo any instance attributes that are None when they should be methods from the class.

        Attributes here refers to the __dataclass_fields__.

        Methods here refers to AsMethod descriptors on any of the super classes of self's class.
        """
        def is_method(cls: type, name: str) -> bool:
            """
            The class has the attribute `name` as a method if:
            - it has the attribute,
            - and it's the AsMethod descriptor.
            """
            return hasattr(cls, name) and isinstance(getattr(cls, name), AsMethod)
        # Check each dataclass attribute.
        for name in self.__dataclass_fields__:
            # If the instance attribute is None
            # and any of the super classes has that as a method,
            # then delete the None instance attribute.
            if (
                getattr(self, name) is None
                and any(is_method(cls, name) for cls in getmro(type(self)))
            ):
                delattr(self, name)


class AsMethod:
    """
    A descriptor for converting function attributes into bound methods.

    To support both inheritance and dataclasses, if the method is None,
    then nothing is set.
    """

    def __init__(self: AsMethod, name: str, default: Callable) -> None:
        if not callable(default):
            raise TypeError(f"'default' must be a method i.e. callable.")
        self.name = name
        self.default = default

    def __get__(self: AsMethod, obj: "Attributes", cls: type) -> Callable:
        # Already has the attribute on the object.
        if self.name in vars(obj):
            return vars(obj)[self.name]
        # Otherwise use the default as a method.
        if next(iter(signature(self.default).parameters), None) in ("self", "ga"):
            return MethodType(self.default, obj)
        # Otherwise use the default as a function.
        return self.default

    def __set__(self: AsMethod, obj: "Attributes", method: Optional[Callable]) -> None:
        if method is None:
            return
        elif not callable(method):
            raise TypeError(f"'{self.name}' must be a method i.e. callable.")
        elif next(iter(signature(method).parameters), None) in ("self", "ga"):
            method = MethodType(method, obj)
        vars(obj)[self.name] = method

    def __delete__(self: AsMethod, obj: "Attributes") -> None:
        del vars(obj)[self.name]


class Attributes(AttributesData):
    """
    The Attributes class inherits default attributes from AttributesData
    and implements methods, descriptors, and properties.

    The built-in methods provide interfacing to the database.
    >>> ga.save_population()  # references ga.database.insert_current_population(ga)
    The descriptors are used to convert function attributes into methods.
    >>> ga.gene_impl = lambda self: ...  # self is turned into an implicit argument.
    The properties are used to validate certain inputs.

    Developer Notes:

        If inherited, the descriptors may be overridden with a method implementation,
        but this removes the descriptor.

        To override default attributes, we recommend creating a dataclass inheriting AttributesData.
        Then inherit the Attributes and AttributesDataSubclass, in that order.
        >>> from dataclasses import dataclass
        >>> @dataclass
        >>> class MyDefaults(AttributesData):
        ...     run: int = 10
        ... 
        >>> class MyAttributes(Attributes, MyDefaults):
        ...     pass
        ... 
    """

    #============================#
    # Built-in database methods: #
    #============================#

    def save_population(self: Attributes) -> None:
        """Saves the current population to the database."""
        self.database.insert_current_population(self)

    def save_chromosome(self: Attributes, chromosome: Chromosome) -> None:
        """
        Saves a chromosome to the database.

        Parameters
        ----------
        chromosome : Chromosome
            The chromosome to be saved.
        """
        self.database.insert_current_chromosome(self.current_generation, chromosome)

    #===========================#
    # Descriptors which convert #
    # functions into methods:   #
    #===========================#

    fitness_function_impl = AsMethod("fitness_function_impl", Fitness.is_it_5)
    make_gene = AsMethod("make_gene", Gene)
    make_chromosome = AsMethod("make_chromosome", Chromosome)
    make_population = AsMethod("make_population", Population)
    gene_impl = AsMethod("gene_impl", rand_1_to_10)
    chromosome_impl = AsMethod("chromosome_impl", use_genes)
    population_impl = AsMethod("population_impl", use_chromosomes)
    dist = AsMethod("dist", dist_fitness)
    weighted_random = AsMethod("weighted_random", simple_linear)
    parent_selection_impl = AsMethod("parent_selection_impl", Parent.Rank.tournament)
    crossover_individual_impl = AsMethod("crossover_individual_impl", Crossover.Individual.single_point)
    crossover_population_impl = AsMethod("crossover_population_impl", Crossover.Population.sequential)
    survivor_selection_impl = AsMethod("survivor_selection_impl", Survivor.fill_in_best)
    mutation_individual_impl = AsMethod("mutation_individual_impl", Mutation.Individual.individual_genes)
    mutation_population_impl = AsMethod("mutation_population_impl", Mutation.Population.random_avoid_best)
    termination_impl = AsMethod("termination_impl", Termination.fitness_generation_tolerance)

    #=============#
    # Properties: #
    #=============#

    @property
    def run(self: AttributesProperties) -> int:
        return vars(self)["run"]

    @run.setter
    def run(self: AttributesProperties, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError("ga.run counter must be an integer greater than or equal to 0.")
        vars(self)["run"] = value

    @property
    def current_generation(self: AttributesProperties) -> int:
        return vars(self)["current_generation"]

    @current_generation.setter
    def current_generation(self: AttributesProperties, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError("ga.current_generation must be an integer greater than or equal to 0")
        vars(self)["current_generation"] = value

    @property
    def chromosome_length(self: AttributesProperties) -> int:
        return vars(self)["chromosome_length"]

    @chromosome_length.setter
    def chromosome_length(self: AttributesProperties, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("ga.chromosome_length must be an integer greater than and not equal to 0.")
        vars(self)["chromosome_length"] = value

    @property
    def population_size(self: AttributesProperties) -> int:
        return vars(self)["population_size"]

    @population_size.setter
    def population_size(self: AttributesProperties, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("ga.population_size must be an integer greater than and not equal to 0.")
        vars(self)["population_size"] = value

    @property
    def max_chromosome_mutation_rate(self: AttributesProperties) -> float:
        # Default value.
        if vars(self).get("max_chromosome_mutation_rate", None) is None:
            return min(self.chromosome_mutation_rate * 2, (self.chromosome_mutation_rate + 1) / 2)
        # Set value.
        return vars(self)["max_chromosome_mutation_rate"]

    @max_chromosome_mutation_rate.setter
    def max_chromosome_mutation_rate(self: AttributesProperties, value: Optional[float]) -> None:
        # Use default or a valid float.
        if value is None or (isinstance(value, (float, int)) and 0 <= value <= 1):
            vars(self)["max_chromosome_mutation_rate"] = value
        else:
            raise ValueError("Max chromosome mutation rate must be between 0 and 1")

    @property
    def min_chromosome_mutation_rate(self: AttributesProperties) -> float:
        # Default value.
        if vars(self).get("min_chromosome_mutation_rate", None) is None:
            return max(self.chromosome_mutation_rate / 2, self.chromosome_mutation_rate * 2 - 1)
        # Set value.
        return vars(self)["min_chromosome_mutation_rate"]

    @min_chromosome_mutation_rate.setter
    def min_chromosome_mutation_rate(self: AttributesProperties, value: Optional[float]) -> None:
        # Use default or a valid float.
        if value is None or (isinstance(value, (float, int)) and 0 <= value <= 1):
            vars(self)["min_chromosome_mutation_rate"] = value
        else:
            raise ValueError("Min chromosome mutation rate must be between 0 and 1")

    @property
    def database_name(self: AttributesProperties) -> str:
        return vars(self)["database_name"]

    @database_name.setter
    def database_name(self: AttributesProperties, name: str) -> None:
        # Update the database's name.
        self.database._database_name = name
        # Set the attribute for itself.
        vars(self)["database_name"] = name

    @property
    def graph(self: AttributesProperties) -> Graph:
        return vars(self)["graph"]

    @graph.setter
    def graph(self: AttributesProperties, graph: Callable[[Database], Graph]) -> None:
        vars(self)["graph"] = graph(self.database)

    @property
    def active(self: AttributesProperties) -> Callable[[], bool]:
        return self.termination_impl
