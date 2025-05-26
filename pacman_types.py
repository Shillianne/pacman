from typing import Optional, Self, Union, Protocol
import numpy as np

Number = Union[int, float]
Value = Union[Number, str]


class GridProtocol(Protocol):
    data: Union[list[list[int]], np.ndarray]
    width: int
    height: int
    def __init__(self, width: int, height: int, initialValue: bool = False, bitRepresentation=None):
        ...

    def copy(self) -> Self:
        ...

    def shallowCopy(self) -> Self:
        ...

    def deepCopy(self) -> Self:
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[list[int], list[list[int]]]:
        ...

    def count(self) -> int:
        ...

    def asList(self, key: bool = True) -> list[tuple[int, int]]:
        ...


class LayoutProtocol(Protocol):
    width: int
    height: int
    walls: GridProtocol
    food: GridProtocol
    capsules: list[tuple[Number, Number]]
    agentPositions: list[tuple[int, tuple[Number, Number]]]
    numGhosts: int
    layoutText: str
    totalFood: int

    def __init__(self, layoutText: str):
        ...

    def deepCopy(self) -> Self:
        ...


class ConfigurationProtocol(Protocol):
    pos: tuple[Number, Number]
    direction: str

    def generateSuccessor(self, vector: tuple[Number, Number]) -> Self:
        ...

    def getPosition(self) -> tuple[Number, Number]:
        ...

    def getDirection(self) -> str:
        ...


class AgentStateProtocol(Protocol):
    scaredTimer: int
    configuration: ConfigurationProtocol
    isPacman: bool
    start: ConfigurationProtocol

    def copy(self) -> Self:
        ...

    def getPosition(self) -> tuple[Number, Number]:
        ...


class GameStateDataProtocol(Protocol):
    layout: LayoutProtocol
    food: GridProtocol
    agentStates: list[AgentStateProtocol]


class GameStateProtocol(Protocol):
    """
    A class that defines the basic protocol the actual GameState class found in pacman.py is supposed to have.
    This class exists for the purpose of implementing type checking and for making coding easier.
    """
    data: GameStateDataProtocol

    def __init__(self, prevState: Optional[Self] = None):
        ...
    
    def getNumFood(self) -> int:
        ...

    def deepCopy(self) -> Self:
        ...

    def generateSuccessor(self, agentIndex: int, action: str) -> Self:
        ...


class GameProtocol(Protocol):
    state: GameStateProtocol


class Seed:
    value: int = 42

    @classmethod
    def set_seed(cls, value: int):
        Seed.value = value

    @classmethod
    def get_value(cls):
        return Seed.value
