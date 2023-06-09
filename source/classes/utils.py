from typing import List

def index_finder(_list: List, value : int) -> List:
    return [ind for ind, val in enumerate(_list) if val == value]


def all_equal(_list: List) -> bool:
    _list = iter(_list)
    try:
        first = next(_list)
    except StopIteration:
        return True
    return all(first == x for x in _list)


def milliseconds_to_minutes_seconds(milliseconds: float) -> tuple:
    minutes = (milliseconds // (1000 * 60)) % 60
    seconds = (milliseconds // 1000) % 60
    return (minutes, seconds)