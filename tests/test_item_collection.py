import pytest

from aptrade.metabase import ItemCollection


class Dummy:
    pass


def test_append_with_name_registers_item_and_attribute():
    collection = ItemCollection()
    first = Dummy()
    second = Dummy()

    collection.append(first, "first")
    collection.append(second, "second")

    assert len(collection) == 2
    assert collection[0] is first
    assert collection[1] is second
    assert collection.first is first
    assert collection.second is second
    assert collection.getnames() == ["first", "second"]
    assert list(collection.getitems()) == [("first", first), ("second", second)]
    assert collection.getbyname("first") is first
    assert collection.getbyname("second") is second


def test_getbyname_missing_raises_value_error():
    collection = ItemCollection()
    collection.append(Dummy(), "present")

    with pytest.raises(ValueError):
        collection.getbyname("absent")
