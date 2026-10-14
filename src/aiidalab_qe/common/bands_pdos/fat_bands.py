"""Outlines of the `fat bands`, i.e. bands whose thickness shows a projection.

The thickness of a fat band has to be measured perpendicular to the band, otherwise
steep bands look thin regardless of their projection. As the two axes of the plot have
different units, "perpendicular" is only meaningful on screen. The outlines are
therefore constructed in pixels, using the scales of the plot, and converted back to
data coordinates afterwards. Points are stored as complex numbers `x + iy`.

The outline of a band is the envelope of discs centred on the band, whose radius is
the local half-width. At a bend, the outer side goes around the arc of the disc with
one point per `ARC_STEP_ANGLE` of the turn (a single point, the miter join, for gentle
bends), while the inner side meets at a single point.

The outlines are returned as NaN-separated polygons, to be plotted in a single trace
with `fill="toself"`. WebGL fills every polygon on its own and cannot handle
self-intersecting ones, which is why the outline is split into pieces that share their
edges (overlapping pieces would show darker seams with a transparent fill): strips of
regular segments, fans around the outer side of sharp bends (as part of a strip, their
inner point often ends up on the wrong side of a neighbouring piece) and folded
segments, which occur where the band bends tighter than its thickness, on their own.
"""

import numpy as np

# Angular resolution of the outer side of bends, bends up to this angle get a miter join
ARC_STEP_ANGLE = np.radians(30)
# Largest change of the half-width along the band that is followed by the envelope
MAX_SLOPE = 0.7

_NAN = complex(np.nan, np.nan)


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
    first, last = _visible_runs(x, y, halfwidth, min_halfwidth / scale_y, y_window)
    if first.size == 0:
        return np.array([]), np.array([])

    # Work in pixels on the flattened bands: vertex `i` and `i + 1` are connected by
    # segment `i`, the segments between consecutive bands are never used
    num_bands = y.shape[0]
    p = np.tile(np.asarray(x, dtype=float) * scale_x, num_bands) + 1j * (
        y.ravel() * scale_y
    )
    radius = halfwidth.ravel() * scale_y
    upper, lower, run, vertex = _offset_sides(p, radius, first, last)

    # Pieces between consecutive points of the same run
    piece = np.flatnonzero(run[:-1] == run[1:])
    is_fan = vertex[piece] == vertex[piece + 1]
    is_folded = ~is_fan & _is_folded(
        upper, lower, piece, _unit(np.diff(p)[vertex[piece]])
    )

    # Merge consecutive pieces of the same kind (regular or fan) into strips, folded
    # ones are left out and leave a gap in `piece`, as do the boundaries between runs
    regular, fan = piece[~is_folded], is_fan[~is_folded]
    new_strip = np.r_[True, (np.diff(regular) != 1) | (fan[1:] != fan[:-1])]
    starts = np.flatnonzero(new_strip)
    ends = np.r_[starts[1:], regular.size] - 1
    strips = _strip_polygons(upper, lower, regular[starts], regular[ends] + 1)

    poly = np.concatenate([strips, _folded_polygons(upper, lower, piece[is_folded])])
    return poly.real / scale_x, poly.imag / scale_y


def _offset_sides(p, radius, first, last):
    """Return the points of the upper and lower side of the runs of segments
    `first..last`, with the run and the vertex each point belongs to.

    Sharp bends contribute several points along the arc of their outer side, all
    other vertices one.
    """
    # Each run covers the vertices `first..last + 1`
    num_vertices = last - first + 2
    run, pos = _ragged_arange(num_vertices)
    vertex = first[run] + pos
    is_start = pos == 0
    is_end = pos == num_vertices[run] - 1

    # Segments before and after each vertex, the ends of a run only have one
    seg_prev = np.where(is_start, vertex, vertex - 1)
    seg_next = np.where(is_end, vertex - 1, vertex)
    d = np.diff(p)
    length = np.abs(d)
    t_prev, t_next = _unit(d[seg_prev]), _unit(d[seg_next])
    # Signed turning angle, zero at the ends
    turn = np.arctan2(_cross(t_prev, t_next), _dot(t_prev, t_next))
    # Normals point to the upper side, the bisector is the normal at the ends
    bisector = _unit(1j * (t_prev + t_next))
    r = radius[vertex]
    # The outer side of a bend is the upper one for a clockwise turn
    outer_side = np.where(turn > 0, -1.0, 1.0)

    # Largest offset on the inner side of a bend, relative to the half-width, for
    # which the inner side does not fold over itself. Beyond the local radius of
    # curvature, the offsets of neighbouring vertices cross each other and the
    # overlapping pieces would be drawn twice.
    available = np.minimum(length[seg_prev], length[seg_next]) / 2
    with np.errstate(divide="ignore"):
        curvature_radius = available / np.tan(np.abs(turn) / 2)
    inner_limit = np.minimum(1.0, curvature_radius / np.maximum(r, 1e-12))

    # Where the radius changes along the band, the envelope of the discs touches each
    # disc slightly behind its centre, by the rate of change `slope`. This is the same
    # as touching a smaller disc whose centre is shifted back along the tangent.
    span = length[seg_prev] + np.where(is_start | is_end, 0.0, length[seg_next])
    slope = (radius[seg_next + 1] - radius[seg_prev]) / span
    slope = np.clip(slope, -MAX_SLOPE, MAX_SLOPE)
    centre = p[vertex] + 1j * slope * r * bisector
    r = np.sqrt(1 - slope**2) * r

    # Inner side: a single point on the bisector (miter), limited as above
    inner = (
        centre
        - outer_side * r * inner_limit / np.maximum(np.cos(turn / 2), 1e-3) * bisector
    )

    # Outer side: the polygon around the arc between the normals of the two segments,
    # one point per `ARC_STEP_ANGLE` of the turn. For a single point, this is the miter
    # join, at the ends of a run the normal of the segment.
    num_points = np.maximum(1, np.ceil(np.abs(turn) / ARC_STEP_ANGLE)).astype(int)
    owner, step = _ragged_arange(num_points)
    half_step = turn[owner] / (2 * num_points[owner])
    direction = 1j * t_prev[owner] * np.exp(1j * (2 * step + 1) * half_step)
    outer = centre[owner] + outer_side[owner] * r[owner] / np.cos(half_step) * direction
    inner = inner[owner]

    upper_is_outer = outer_side[owner] > 0
    upper = np.where(upper_is_outer, outer, inner)
    lower = np.where(upper_is_outer, inner, outer)
    return upper, lower, run[owner], vertex[owner]


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


def _is_folded(upper, lower, piece, tangent):
    """Return whether the pieces between the points `piece` and `piece + 1` fold over,
    i.e. a side runs backwards along `tangent`, or the sides cross each other."""
    upper_a, upper_b = upper[piece], upper[piece + 1]
    lower_a, lower_b = lower[piece], lower[piece + 1]

    def backwards(a, b):
        return _dot(b - a, tangent) < 0

    return (
        backwards(upper_a, upper_b)
        | backwards(lower_a, lower_b)
        | _crossing(upper_a, upper_b, lower_a, lower_b)
        | _crossing(upper_a, lower_a, upper_b, lower_b)
    )


def _strip_polygons(upper, lower, starts, ends):
    """Return the polygons along the upper side of the points `starts..ends`, back
    along the lower side, each followed by a NaN separator."""
    num_points = ends - starts + 1
    polygon, pos = _ragged_arange(2 * num_points + 1)
    start, num_points = starts[polygon], num_points[polygon]
    # Index into the upper points, followed by the lower ones and the separator
    point = np.where(
        pos < num_points, start + pos, upper.size + start + 2 * num_points - 1 - pos
    )
    point[pos == 2 * num_points] = 2 * upper.size
    return np.concatenate([upper, lower, [_NAN]])[point]


def _folded_polygons(upper, lower, folded):
    """Return the polygons of the folded pieces, each followed by a NaN separator.

    A folded piece crosses itself. Its corners all lie within the half-width of its
    segment, so ordering them by angle around their centre gives a simple polygon
    that still stays within the outline of the band.
    """
    corners = np.column_stack(
        [upper[folded], upper[folded + 1], lower[folded + 1], lower[folded]]
    )
    order = np.argsort(np.angle(corners - corners.mean(axis=1, keepdims=True)), axis=1)
    separator = np.full((folded.size, 1), _NAN)
    return np.hstack([np.take_along_axis(corners, order, axis=1), separator]).ravel()


def _ragged_arange(counts):
    """Return, for groups of `counts` elements, the group of each element and its
    position within the group."""
    group = np.repeat(np.arange(counts.size), counts)
    return group, np.arange(group.size) - (np.cumsum(counts) - counts)[group]


def _crossing(a, b, c, d):
    """Return whether the segments `a-b` and `c-d` cross."""

    def side(o, p, q):
        return _cross(p - o, q - o)

    return (side(a, b, c) * side(a, b, d) < 0) & (side(c, d, a) * side(c, d, b) < 0)


# Written out, as complex multiplication does not give exact zeros for parallel vectors
def _cross(a, b):
    return a.real * b.imag - a.imag * b.real


def _dot(a, b):
    return a.real * b.real + a.imag * b.imag


def _unit(v):
    norm = np.abs(v)
    norm[norm < 1e-12] = 1.0
    return v / norm
