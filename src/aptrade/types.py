from typing import Any, Self


class Percentage(float):
    """Validated float representing a fraction between 0 and 1."""

    def __new__(cls, value: Any) -> Self:
        val = float(value)
        if not 0.0 <= val <= 1.0:
            raise ValueError("Percentage must be between 0 and 1")
        return super().__new__(cls, val)

    @classmethod
    def from_percent(cls, value: Any) -> Self:
        return cls(float(value) / 100.0)
