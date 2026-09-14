"""Versioned authoring interchange for the OKBE and WarpCraft grid editors.

The world is the tutorial's authoring model, not a compiled transition kernel.
Runtime adapters must explicitly distinguish preserved data from executed rules.
"""
from copy import deepcopy
import math

FORMAT = "okbe-warpcraft-level"
VERSION = 1


def read_level(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("level must be a JSON object")
    if "format" in record and (record["format"] != FORMAT or record.get("format_version") != VERSION):
        raise ValueError("unsupported level format/version")
    if record.get("coordinates", "row-major-top-left") != "row-major-top-left":
        raise ValueError("unsupported coordinate convention")
    world = record.get("world")
    if not isinstance(world, dict):
        raise ValueError("level must contain a world object")
    def integer(value, low, high, label):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    for name in ("rows", "cols"):
        integer(world.get(name), 2, 40, name)
    count = world["rows"] * world["cols"]
    def cell(value):
        integer(value, 0, count - 1, "cell")
    for field in ("walls", "constraints"):
        if not isinstance(world.get(field, []), list):
            raise ValueError(f"{field} must be an array")
        for value in world.get(field, []):
            cell(value)
    if world.get("start") is not None:
        cell(world["start"])
    for group in world.get("goals", []):
        integer(group["id"], 1, 1_000_000, "goal group ID")
        for entry in group["cells"]:
            cell(entry if type(entry) is int else entry["li"])
            if isinstance(entry, dict) and entry.get("a") is not None:
                integer(entry["a"], 0, 5, "goal action")
    for field, types in (("internal_goalstates_2_type_ind", range(6)),
                         ("boolean_states_2_type_ind", (1, 2, 3, 5, 6))):
        for key, value in world.get(field, {}).items():
            cell(int(key))
            if type(value) is not int or value not in types:
                raise ValueError(f"unknown {field} type {value}")
    noise = world.get("p_main", 1)
    if not isinstance(noise, (int, float)) or not math.isfinite(noise) or not 0 < noise <= 1:
        raise ValueError("p_main must be in (0, 1]")
    result = deepcopy(record)
    result.update(format=FORMAT, format_version=VERSION, coordinates="row-major-top-left")
    result.setdefault("strings", {})
    result.setdefault("name", "Untitled")
    return result


def make_level(world: dict, **metadata) -> dict:
    return read_level({**metadata, "world": world})
