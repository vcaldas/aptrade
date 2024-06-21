def calculate(num: int) -> int:
    """Calculate the given fibonacci number

    Args:
        num: The fibonacci number you want to generate

    Returns:
        The requested fibonacci number
    """
    a, b = 0, 1
    for _ in range(num):
        a, b = b, a + b
    return a
