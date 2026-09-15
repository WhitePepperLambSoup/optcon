"""Annotate an existing signature, get checked units at the boundary.

The rule this module follows - the same one prysm and poppy already proved
works in production optical code - is **units at the boundary, plain numbers
inside**:

* each declared argument is converted to its declared unit, and the function
  body receives the bare magnitude, so existing arithmetic keeps working;
* each declared return value is interpreted in its declared unit, and a
  Quantity returned with the wrong dimension or amplitude order is rejected.

That makes the cost of adoption a signature change rather than a rewrite.
"""

from __future__ import annotations

import functools
import inspect
import typing
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, Callable, get_args, get_origin, get_type_hints

from .errors import AmplitudeOrderError, DimensionError, UnitError
from .quantity import Quantity, q
from .units import Unit
from .units import unit as _unit

#: Field-name suffixes that carry a unit, checked longest first so that
#: ``_mrad`` wins over ``_m`` and ``_mm`` over ``_m``.
_FIELD_SUFFIXES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            ("_mrad", "mrad"), ("_urad", "urad"), ("_nrad", "nrad"),
            ("_deg", "deg"), ("_rad", "rad"),
            ("_mm", "mm"), ("_nm", "nm"), ("_um", "um"), ("_cm", "cm"), ("_m", "m"),
            ("_ms", "ms"), ("_us", "us"), ("_ns", "ns"), ("_ps", "ps"), ("_fs", "fs"),
            ("_s", "s"),
            ("_mW", "mW"), ("_mW_cm2", "mW/cm^2"), ("_w_cm2", "W/cm^2"),
            ("_W", "W"), ("_uW", "uW"), ("_w", "W"),
            ("_mj", "mJ"), ("_uj", "uJ"), ("_J", "J"), ("_j", "J"),
            ("_THz", "THz"), ("_GHz", "GHz"), ("_MHz", "MHz"), ("_kHz", "kHz"),
            ("_Hz", "Hz"), ("_hz", "Hz"),
            ("_eV", "eV"), ("_meV", "meV"),
            ("_mK", "mK"), ("_K", "K"),
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def _resolve_annotation(annotation: Any) -> tuple[Unit | None, int | None]:
    """Pull ``(unit, amp_order)`` out of an ``Annotated`` hint, if present."""
    if get_origin(annotation) is not typing.Annotated:
        return None, None
    metadata = get_args(annotation)[1:]
    resolved_unit: Unit | None = None
    order: int | None = None
    for item in metadata:
        if isinstance(item, Unit):
            resolved_unit = item
        elif isinstance(item, str):
            resolved_unit = _unit(item)
        elif isinstance(item, int) and not isinstance(item, bool):
            order = int(item)
        else:
            raise UnitError(
                f"unsupported annotation metadata {item!r}; use a unit string, "
                "a Unit, or an integer amplitude order"
            )
    return resolved_unit, order


def _coerce(value: Any, resolved_unit: Unit | None, order: int | None, context: str):
    """Bring ``value`` onto the declared unit/order, or explain why not."""
    if isinstance(value, Quantity):
        if resolved_unit is not None:
            if value.unit.dimension != resolved_unit.dimension:
                raise DimensionError(
                    f"{context}: expected {resolved_unit}, got {value.unit}"
                )
            value = value.to(resolved_unit)
        if order is not None:
            if value.amp_order is not None and value.amp_order != order:
                raise AmplitudeOrderError(
                    f"{context}: expected amplitude order {order}, got {value.amp_order}"
                )
            value = Quantity(value.value, value.unit, order)
        return value
    if resolved_unit is not None:
        return Quantity(value, resolved_unit, order)
    if order is not None:
        return Quantity(value, _unit("1"), order)
    return value


def checked(function: Callable[..., Any]) -> Callable[..., Any]:
    """Check declared units on the way in and on the way out.

    The wrapped function still receives ordinary numbers in the declared
    units, so its body does not have to change.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        hints = _hints(function)
        signature = _signature(function)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in list(bound.arguments.items()):
            resolved_unit, order = hints.get(name, (None, None))
            if resolved_unit is None and order is None:
                continue
            checked_value = _coerce(value, resolved_unit, order, f"{function.__name__}({name})")
            bound.arguments[name] = checked_value.value

        result = function(*bound.args, **bound.kwargs)

        return_spec = hints.get("return", (None, None))
        if (
            isinstance(return_spec, list)
            and isinstance(result, tuple)
            and len(return_spec) == len(result)
        ):
            return tuple(
                _coerce(item, u, o, f"{function.__name__}() return[{i}]")
                for i, (item, (u, o)) in enumerate(zip(result, return_spec, strict=True))
            )
        if isinstance(return_spec, tuple):
            return_unit, return_order = return_spec
            if return_unit is None and return_order is None:
                return result
            return _coerce(result, return_unit, return_order, f"{function.__name__}() return")
        return result

    return wrapper


def _resolve_hint(hint: Any) -> Any:
    if get_origin(hint) is tuple:
        args = get_args(hint)
        if args and args[-1] is not Ellipsis:
            resolved = [_resolve_annotation(arg) for arg in args]
            if any(u is not None or o is not None for u, o in resolved):
                return resolved
    return _resolve_annotation(hint)


@functools.lru_cache(maxsize=None)
def _hints(function: Callable[..., Any]) -> dict[str, Any]:
    try:
        raw = get_type_hints(function, include_extras=True)
    except Exception:  # noqa: BLE001 - fall back to raw annotations
        raw = dict(getattr(function, "__annotations__", {}))
    return {name: _resolve_hint(hint) for name, hint in raw.items()}


_signature = inspect.signature


def units_of_dataclass(cls: type) -> dict[str, Unit | None]:
    """Infer a unit for each field from its name suffix.

    ``L1_mm`` -> millimetre, ``wavelength_nm`` -> nanometre, ``beta`` ->
    ``None``.  This is the cheapest way to bring a physics dataclass under
    check without touching how anyone writes it.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    inferred: dict[str, Unit | None] = {}
    for field in dataclass_fields(cls):
        inferred[field.name] = _unit_from_name(field.name)
    return inferred


def _unit_from_name(name: str) -> Unit | None:
    for suffix, expression in _FIELD_SUFFIXES:
        if name.endswith(suffix):
            return _unit(expression)
    return None


def to_quantities(instance: Any) -> dict[str, Quantity]:
    """Turn a physics dataclass instance into typed quantities."""
    units = units_of_dataclass(type(instance))
    return {
        name: q(getattr(instance, name), resolved)
        for name, resolved in units.items()
        if resolved is not None
    }


__all__ = ["checked", "to_quantities", "units_of_dataclass"]
