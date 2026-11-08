import csv
import io
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from craterslab.craters import OBSERVABLES, Surface
from craterslab.profiles import Profile


def write_observables_csv(path: Path | str, surfaces: Mapping[str, Surface]) -> None:
    symbols = [definition.symbol for definition in OBSERVABLES]
    rows = [
        [name, surface.type.name, *_observable_cells(surface, symbols)]
        for name, surface in surfaces.items()
    ]
    _write_csv(path, ["name", "type", *symbols], rows)


def write_profile_csv(path: Path | str, profile: Profile) -> None:
    rows = zip(profile.s.tolist(), profile.h.tolist(), strict=True)
    _write_csv(path, ["distance", "height"], rows)


def _observable_cells(surface: Surface, symbols: Sequence[str]) -> list[float | str]:
    return [
        surface.observables[symbol].value if symbol in surface.observables else ""
        for symbol in symbols
    ]


def _write_csv(path: Path | str, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    Path(path).write_text(buffer.getvalue(), encoding="utf-8")
