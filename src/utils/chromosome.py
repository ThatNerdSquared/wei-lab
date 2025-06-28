from random import randint
from typing import Optional
from uuid import uuid4


class Chromosome:
    def __init__(self, length: int = 100):
        self.id = uuid4()
        self.length: int = length
        self._crossovers: dict[int, bool] = {}

    def __str__(self) -> str:
        res = ['-'] * self.length
        for co_loc in self._crossovers.keys():
            res[co_loc] = 'x'
        return ''.join(res)

    def insert_crossover(self, location: int):
        if location > self.length:
            raise ValueError(
                f'Location is out of bounds for {self.length}-bin chromosome!'
            )
        self._crossovers[location] = True

    def randomly_add_crossovers(self, num: int):
        for _ in range(num):
            self.insert_crossover(randint(0, self.length - 1))

    def get_num_crossovers(self) -> int:
        return len(self._crossovers)

    def get_crossover_locations(self) -> list[int]:
        return sorted(self._crossovers.keys())

    def has_crossover_at(self, location: int) -> bool:
        try:
            return self._crossovers[location]
        except KeyError:
            return False

    def has_visible_crossover(self, start: int = 0, end: Optional[int] = None) -> bool:
        endpoint: int = end if end is not None else self.length
        crossover_locations = sorted(self._crossovers.keys())
        num_crossovers_in_region = len(
            [x for x in crossover_locations if x >= start and x <= endpoint]
        )
        return num_crossovers_in_region % 2 == 1
