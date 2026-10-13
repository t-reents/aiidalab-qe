"""Outlines of the `fat bands`, i.e. bands whose thickness shows a projection.

The thickness of a fat band has to be measured perpendicular to the band, otherwise
steep bands look thin regardless of their projection. As the two axes of the plot have
different units, "perpendicular" is only meaningful on screen. The outlines are
therefore constructed in pixels, using the scales of the plot, and converted back to
data coordinates afterwards.

The outline of a band is the envelope of discs centred on the band, whose radius is
the local half-width. At smooth bends, the two sides are offset along the bisector of
the adjacent normals (miter join). At sharp bends, e.g. band crossings, the outer side
follows an arc while the inner side meets at a single point (round join).

The outlines are returned as NaN-separated polygons, to be plotted in a single trace
with `fill="toself"`. WebGL fills every polygon on its own and cannot handle
self-intersecting ones, which is why the outline is split into pieces that share their
edges: strips of regular segments, fans for the round joins and folded segments, which
occur where the band bends tighter than its thickness, on their own.
"""

import numpy as np

# Bends sharper than this get a round outer corner instead of a miter join
ROUND_JOIN_ANGLE = np.radians(30)
# Angular resolution of the arcs of round joins
ARC_STEP_ANGLE = np.radians(30)
# Longest miter, relative to the half-width
MAX_MITER = 2.0
# Largest change of the half-width along the band that is followed by the envelope
MAX_SLOPE = 0.7


def fat_band_polygons(x, y, halfwidth, scale, min_halfwidth=0.0, y_window=None):
    """Return the outlines of the fat bands as NaN-separated polygons.

    :param x: positions of the k-points along the path, shape (number of kpoints,).
    :param y: energies of the bands, shape (number of bands, number of kpoints).
    :param halfwidth: half of the thickness of the bands, in units of `y`, with the
        same shape as `y`.
    :param scale: pixels per unit of `x` and of `y` in the plot.
    :param min_halfwidth: segments thinner than this, in pixels, at both of their ends
        are not drawn.
    :param y_window: if given, only the segments whose fat band reaches into this
        (lower, upper) range of `y` are drawn.
    :return: the x and y coordinates of the polygons in data units.
    """
    scale_x, scale_y = scale
    num_bands = y.shape[0]

    # Work in pixels on the flattened bands: vertex `i` and `i + 1` are connected by
    # segment `i`, the segments between consecutive bands are never used
    px = np.tile(np.asarray(x, dtype=float) * scale_x, num_bands)
    py = y.ravel() * scale_y
    radius = halfwidth.ravel() * scale_y

    first, last = _visible_runs(x, y, halfwidth, min_halfwidth / scale_y, y_window)
    if first.size == 0:
        return np.array([]), np.array([])

    dx, dy = np.diff(px), np.diff(py)
    length = np.hypot(dx, dy)
    length[length == 0] = 1.0
    tx, ty = dx / length, dy / length
    nx, ny = -ty, tx

    # Each run covers the vertices `first..last + 1`
    num_vertices = last - first + 2
    run = np.repeat(np.arange(first.size), num_vertices)
    pos = np.arange(num_vertices.sum()) - np.repeat(
        np.cumsum(num_vertices) - num_vertices, num_vertices
    )
    vertex = first[run] + pos
    is_start = pos == 0
    is_end = pos == num_vertices[run] - 1
    interior = ~is_start & ~is_end

    # Signed turning angle at the interior vertices
    turn = np.zeros(vertex.size)
    v = vertex[interior]
    turn[interior] = np.arctan2(
        tx[v - 1] * ty[v] - ty[v - 1] * tx[v],
        tx[v - 1] * tx[v] + ty[v - 1] * ty[v],
    )
    is_round = interior & (np.abs(turn) > ROUND_JOIN_ANGLE)

    # Round joins are drawn with several points along an arc, others with one
    num_points = np.where(
        is_round, 1 + np.ceil(np.abs(turn) / ARC_STEP_ANGLE).astype(int), 1
    )
    arc_step = np.arange(num_points.sum()) - np.repeat(
        np.cumsum(num_points) - num_points, num_points
    )
    run, vertex, turn = (np.repeat(a, num_points) for a in (run, vertex, turn))
    is_start, is_end, is_round = (
        np.repeat(a, num_points) for a in (is_start, is_end, is_round)
    )
    num_points = np.repeat(num_points, num_points)
    is_smooth = ~is_start & ~is_end & ~is_round

    upper_x, upper_y = np.empty(vertex.size), np.empty(vertex.size)
    lower_x, lower_y = np.empty(vertex.size), np.empty(vertex.size)

    def offset(mask, normal_x, normal_y, slope, upper=1.0, lower=1.0):
        """Offset the vertices to both sides, on the envelope of the discs.

        Where the radius changes along the band, the envelope touches the disc
        slightly behind its centre, by the rate of change `slope`. `upper` and
        `lower` scale the offset to either side.
        """
        r = radius[vertex[mask]]
        along = -slope * r
        across = np.sqrt(1 - slope**2) * r
        # The tangent is the normal rotated clockwise
        base_x = px[vertex[mask]] + along * normal_y
        base_y = py[vertex[mask]] - along * normal_x
        upper_x[mask] = base_x + upper * across * normal_x
        upper_y[mask] = base_y + upper * across * normal_y
        lower_x[mask] = base_x - lower * across * normal_x
        lower_y[mask] = base_y - lower * across * normal_y

    def inner_limit(mask):
        """Return the largest offset on the inner side of a bend, relative to the
        half-width, for which the inner side does not fold over itself.

        Beyond the local radius of curvature, the offsets of neighbouring vertices
        cross each other and the overlapping pieces would be drawn twice.
        """
        v = vertex[mask]
        half_turn = np.abs(turn[mask]) / 2
        available = np.minimum(length[v - 1], length[v]) / 2
        with np.errstate(divide="ignore"):
            curvature_radius = available / np.tan(half_turn)
        return np.minimum(1.0, curvature_radius / np.maximum(radius[v], 1e-12))

    # Ends of a run: perpendicular to their only segment
    seg = vertex[is_start]
    offset(
        is_start, nx[seg], ny[seg], _clip((radius[seg + 1] - radius[seg]) / length[seg])
    )
    seg = vertex[is_end] - 1
    offset(
        is_end, nx[seg], ny[seg], _clip((radius[seg + 1] - radius[seg]) / length[seg])
    )

    # Smooth bends: miter join along the bisector of the adjacent normals
    v = vertex[is_smooth]
    bisector_x, bisector_y = _normalize(nx[v - 1] + nx[v], ny[v - 1] + ny[v])
    rate = (radius[v + 1] - radius[v - 1]) / (length[v - 1] + length[v])
    miter = 1 / np.maximum(np.cos(turn[is_smooth] / 2), 1 / MAX_MITER)
    # The inner side is on the left (upper) for a counterclockwise turn
    inner = miter * inner_limit(is_smooth)
    upper_is_inner = turn[is_smooth] > 0
    offset(
        is_smooth,
        bisector_x,
        bisector_y,
        _clip(rate),
        upper=np.where(upper_is_inner, inner, miter),
        lower=np.where(upper_is_inner, miter, inner),
    )

    # Sharp bends: arc on the outer side, a single (limited) miter point on the inner
    v = vertex[is_round]
    r = radius[v]
    angle = np.arctan2(ny[v - 1], nx[v - 1]) + turn[is_round] * arc_step[is_round] / (
        num_points[is_round] - 1
    )
    outer_side = -np.sign(turn[is_round])
    outer_x = px[v] + outer_side * r * np.cos(angle)
    outer_y = py[v] + outer_side * r * np.sin(angle)
    bisector_x, bisector_y = _normalize(nx[v - 1] + nx[v], ny[v - 1] + ny[v])
    inner_distance = (
        r * inner_limit(is_round) / np.maximum(np.cos(turn[is_round] / 2), 1e-3)
    )
    inner_x = px[v] - outer_side * inner_distance * bisector_x
    inner_y = py[v] - outer_side * inner_distance * bisector_y
    upper_is_outer = outer_side > 0
    upper_x[is_round] = np.where(upper_is_outer, outer_x, inner_x)
    upper_y[is_round] = np.where(upper_is_outer, outer_y, inner_y)
    lower_x[is_round] = np.where(upper_is_outer, inner_x, outer_x)
    lower_y[is_round] = np.where(upper_is_outer, inner_y, outer_y)

    # Pieces between consecutive points of the same run
    piece = np.flatnonzero(run[:-1] == run[1:])
    is_fan = vertex[piece] == vertex[piece + 1]
    seg = vertex[piece]
    upper_a = (upper_x[piece], upper_y[piece])
    upper_b = (upper_x[piece + 1], upper_y[piece + 1])
    lower_a = (lower_x[piece], lower_y[piece])
    lower_b = (lower_x[piece + 1], lower_y[piece + 1])
    # A side running backwards, or sides crossing each other (twisted)
    is_folded = ~is_fan & (
        ((upper_b[0] - upper_a[0]) * tx[seg] + (upper_b[1] - upper_a[1]) * ty[seg] < 0)
        | (
            (lower_b[0] - lower_a[0]) * tx[seg] + (lower_b[1] - lower_a[1]) * ty[seg]
            < 0
        )
        | _crossing(upper_a, upper_b, lower_a, lower_b)
        | _crossing(upper_a, lower_a, upper_b, lower_b)
    )
    kind = np.where(is_fan, 2, np.where(is_folded, 1, 0))
    # Merge consecutive pieces of the same kind into one polygon, except folded ones
    new_polygon = np.ones(piece.size, dtype=bool)
    new_polygon[1:] = (
        (run[piece[1:]] != run[piece[:-1]]) | (kind[1:] != kind[:-1]) | is_folded[1:]
    )
    starts = np.flatnonzero(new_polygon)
    ends = np.r_[starts[1:], piece.size] - 1
    regular = ~is_folded[starts]
    starts, ends = starts[regular], ends[regular]
    strip_x, strip_y = _strip_polygons(
        upper_x, upper_y, lower_x, lower_y, piece[starts], piece[ends] + 1
    )

    # A folded piece crosses itself. Its corners all lie within the half-width of its
    # segment, so ordering them by angle around their centre gives a simple polygon
    # that still stays within the outline of the band.
    folded = piece[is_folded]
    corners_x = np.column_stack(
        [upper_x[folded], upper_x[folded + 1], lower_x[folded + 1], lower_x[folded]]
    )
    corners_y = np.column_stack(
        [upper_y[folded], upper_y[folded + 1], lower_y[folded + 1], lower_y[folded]]
    )
    order = np.argsort(
        np.arctan2(
            corners_y - corners_y.mean(axis=1, keepdims=True),
            corners_x - corners_x.mean(axis=1, keepdims=True),
        ),
        axis=1,
    )
    separator = np.full((folded.size, 1), np.nan)
    folded_x = np.hstack([np.take_along_axis(corners_x, order, axis=1), separator])
    folded_y = np.hstack([np.take_along_axis(corners_y, order, axis=1), separator])

    poly_x = np.concatenate([strip_x, folded_x.ravel()])
    poly_y = np.concatenate([strip_y, folded_y.ravel()])
    return poly_x / scale_x, poly_y / scale_y


def _visible_runs(x, y, halfwidth, min_halfwidth, y_window=None):
    """Return the first and last segment of the runs of visible segments.

    A segment is visible if the band is at least `min_halfwidth` thick at one of its
    ends, it is not a discontinuity of the path (zero-length step in x) and its fat band
    reaches into `y_window`, if given. The indices refer to the flattened bands.
    """
    num_kpoints = halfwidth.shape[1]
    visible = np.maximum(halfwidth[:, :-1], halfwidth[:, 1:]) >= min_halfwidth
    visible &= np.diff(x) > 0
    if y_window is not None:
        top, bottom = y + halfwidth, y - halfwidth
        visible &= np.maximum(top[:, :-1], top[:, 1:]) >= y_window[0]
        visible &= np.minimum(bottom[:, :-1], bottom[:, 1:]) <= y_window[1]
    # Pad each band with `False` so that runs never extend across bands
    change = np.diff(np.pad(visible, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    band, first = np.nonzero(change == 1)
    _, stop = np.nonzero(change == -1)
    return band * num_kpoints + first, band * num_kpoints + stop - 1


def _strip_polygons(upper_x, upper_y, lower_x, lower_y, starts, ends):
    """Return the polygons along the upper side of the points `starts..ends`, back
    along the lower side, each followed by a NaN separator."""
    num_points = ends - starts + 1
    num_total = 2 * num_points + 1
    polygon = np.repeat(np.arange(starts.size), num_total)
    pos = np.arange(num_total.sum()) - np.repeat(
        np.cumsum(num_total) - num_total, num_total
    )
    num_points = num_points[polygon]
    upper = pos < num_points
    separator = pos == 2 * num_points
    point = starts[polygon] + np.where(upper, pos, 2 * num_points - 1 - pos)
    point[separator] = starts[polygon][separator]

    poly_x = np.where(upper, upper_x[point], lower_x[point])
    poly_y = np.where(upper, upper_y[point], lower_y[point])
    poly_x[separator] = np.nan
    poly_y[separator] = np.nan
    return poly_x, poly_y


def _crossing(a, b, c, d):
    """Return whether the segments `a-b` and `c-d` cross, given as (x, y) arrays."""

    def side(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    return (side(a, b, c) * side(a, b, d) < 0) & (side(c, d, a) * side(c, d, b) < 0)


def _normalize(vx, vy):
    norm = np.hypot(vx, vy)
    norm[norm < 1e-12] = 1.0
    return vx / norm, vy / norm


def _clip(slope):
    return np.clip(slope, -MAX_SLOPE, MAX_SLOPE)
