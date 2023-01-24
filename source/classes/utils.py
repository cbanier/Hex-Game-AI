def index_finder(_list, value):
    return [ind for ind, val in enumerate(_list) if val == value]


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)