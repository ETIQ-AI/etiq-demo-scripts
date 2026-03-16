def is_empty_object(value: object) -> bool:
    """Return True when the input should be treated as empty."""
    if value is None:
        return True

    type_hierarchy = [
        (cls.__name__, cls.__module__)
        for cls in value.__class__.__mro__
    ]

    def matches_type(target_name: str, module_prefix: str) -> bool:
        return any(
            class_name == target_name and module_name.startswith(module_prefix)
            for class_name, module_name in type_hierarchy
        )

    if matches_type("DataFrame", "pandas"):
        # Pandas treats a DataFrame as empty when any axis has length 0.
        # Explicitly checking the index covers the "no rows populated" case.
        return bool(value.empty or len(value.index) == 0)

    if matches_type("DataFrame", "pyspark.sql"):
        # Spark DataFrames are empty when they contain no rows.
        try:
            return bool(value.isEmpty())
        except AttributeError:
            return value.limit(1).count() == 0

    if matches_type("RDD", "pyspark"):
        return bool(value.isEmpty())

    if matches_type("DataFrame", "pyspark.pandas"):
        return bool(value.empty or len(value.index) == 0)

    if matches_type("Series", "pyspark.pandas"):
        return len(value) == 0

    if isinstance(value, (str, bytes, bytearray, list, tuple, set, frozenset, dict, range)):
        return len(value) == 0

    try:
        empty_attr = getattr(value, "empty")
    except Exception:
        empty_attr = None
    else:
        try:
            return bool(empty_attr)
        except Exception:
            pass

    if hasattr(value, "__len__"):
        try:
            return len(value) == 0
        except (TypeError, ValueError):
            pass

    return False


def get_empty_objects(objects: list[object]) -> list[object]:
    """Return only the empty objects from the provided list."""
    return [obj for obj in objects if is_empty_object(obj.value)]
