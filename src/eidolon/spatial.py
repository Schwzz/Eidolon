"""Small deterministic spatial grid for local particle queries (no display imports)."""

from math import floor


def nearby_pairs(positions, cell_size):
    """Return sorted unique ID pairs in the same or neighboring grid cells.

    The caller checks exact distances. Rebuilding this bounded grid avoids stale
    neighbors after constraint corrections, grabbing, growth, or graph edits.
    """
    cells = {}
    for identifier, (x, y) in enumerate(positions):
        cells.setdefault((floor(x / cell_size), floor(y / cell_size)), []).append(identifier)
    pairs = []
    for (x, y), members in cells.items():
        for offset, first in enumerate(members):
            pairs.extend((first, second) for second in members[offset + 1:])
        for dx, dy in ((0, 1), (1, -1), (1, 0), (1, 1)):
            for first in members:
                for second in cells.get((x + dx, y + dy), ()):
                    pairs.append((min(first, second), max(first, second)))
    return sorted(pairs)
