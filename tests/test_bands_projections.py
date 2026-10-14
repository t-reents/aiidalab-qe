import numpy as np
import pytest

from aiidalab_qe.common.bands_pdos.bandpdosplotly import BandsPdosPlotly
from aiidalab_qe.common.bands_pdos.fat_bands import fat_band_polygons
from aiidalab_qe.common.bands_pdos.utils import prepare_projected_bands_trace


def _bands_data(x, y):
    """Return minimal bands data with `y` of shape (number of kpoints, number of bands)."""
    return {
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "band_type_idx": np.zeros(np.shape(y)[1], dtype=int),
        "fermi_energy": 0.0,
        "paths": [{"x": list(x), "from": "G", "to": "X"}],
        "pathlabels": [["Γ", "X"], [x[0], x[-1]]],
    }


def _projection(projections, label="Si 3s"):
    return {
        "label": label,
        "color": "#ff0000",
        "projections": np.asarray(projections, dtype=float),
        "spin": 0,
    }


def _polygons(x, y):
    """Split NaN-separated coordinates into a list of (n, 2) polygons."""
    breaks = np.flatnonzero(np.isnan(x))
    return [
        np.column_stack([px, py])[np.isfinite(px)]
        for px, py in zip(np.split(x, breaks), np.split(y, breaks))
        if np.isfinite(px).sum() > 2
    ]


def _distance_to_line(points, origin, direction):
    """Perpendicular distance of points from a line."""
    direction = direction / np.linalg.norm(direction)
    rel = points - origin
    return np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])


def _is_simple(polygon):
    """Check that no two non-adjacent edges of the polygon cross."""
    polygon = polygon[np.isfinite(polygon[:, 0])]
    edges = list(zip(polygon, np.roll(polygon, -1, axis=0)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    for i, (a, b) in enumerate(edges):
        for j in range(i + 2, len(edges)):
            if i == 0 and j == len(edges) - 1:
                continue
            c, d = edges[j]
            if (
                cross(a, b, c) * cross(a, b, d) < -1e-9
                and cross(c, d, a) * cross(c, d, b) < -1e-9
            ):
                return False
    return True


def test_flat_band_matches_vertical_width():
    """For a flat band, the thickness is the same as a vertical offset."""
    x = np.linspace(0, 1, 5)
    y = np.full((1, 5), 2.0)
    halfwidth = np.full((1, 5), 0.25)

    px, py = fat_band_polygons(x, y, halfwidth, scale=(400, 20))

    (polygon,) = _polygons(px, py)
    np.testing.assert_allclose(
        np.sort(np.unique(polygon[:, 1].round(12))), [1.75, 2.25]
    )
    np.testing.assert_allclose([polygon[:, 0].min(), polygon[:, 0].max()], [0, 1])


@pytest.mark.parametrize("slope", [0.0, 1.0, 10.0, 200.0])
@pytest.mark.parametrize("scale", [(400, 20), (100, 50)])
def test_thickness_is_independent_of_slope(slope, scale):
    """On screen, the outline is offset by the half-width perpendicular to the band."""
    x = np.linspace(0, 1, 11)
    y = (slope * x)[None, :]
    halfwidth = np.full_like(y, 0.25)

    px, py = fat_band_polygons(x, y, halfwidth, scale=scale)

    pixels = np.column_stack([px, py])[np.isfinite(px)] * scale
    distance = _distance_to_line(pixels, np.zeros(2), np.array(scale) * [1, slope])
    np.testing.assert_allclose(distance, 0.25 * scale[1])


def test_round_join_at_kink():
    """The outer side of a sharp bend goes around the arc at the half-width from the
    vertex, i.e. its points lie on the circumscribed polygon of the arc."""
    x = np.linspace(0, 2, 21)
    y = (5 * np.abs(x - 1))[None, :]  # V-shaped kink at x = 1
    halfwidth = np.full_like(y, 0.5)
    scale = (100, 20)

    px, py = fat_band_polygons(x, y, halfwidth, scale)

    pixels = np.column_stack([px, py])[np.isfinite(px)] * scale
    distance = np.hypot(pixels[:, 0] - 100, pixels[:, 1])
    below = pixels[:, 1] < 0
    # The 90° bend below the kink is covered in 3 steps of 30°, so the points are
    # 1 / cos(15°) from the vertex
    assert below.sum() > 2
    np.testing.assert_allclose(distance[below], 0.5 * scale[1] / np.cos(np.radians(15)))
    assert all(_is_simple(polygon * scale) for polygon in _polygons(px, py))


def test_polygons_are_simple():
    """Every polygon can be triangulated by WebGL, also at narrow peaks and steps."""
    x = np.linspace(0, 1, 60)
    y = np.vstack(
        [
            3 * np.exp(-(((x - 0.5) / 0.02) ** 2)),  # narrow peak
            10 * np.tanh((x - 0.3) / 0.01),  # nearly vertical step
            np.sin(20 * x),
        ]
    )
    halfwidth = 0.3 * (1 + np.sin(7 * x))[None, :] * np.ones((3, 1))

    px, py = fat_band_polygons(x, y, halfwidth, scale=(450, 21))

    polygons = _polygons(px, py)
    assert polygons
    assert all(_is_simple(polygon * (450, 21)) for polygon in polygons)


def test_skip_thin_stretches_and_path_discontinuities():
    """Thin segments are not drawn and polygons do not bridge a zero-length step."""
    x = np.array([0, 1, 2, 3, 3, 4, 5])
    y = np.zeros((1, 7))
    halfwidth = np.array([[0, 0, 1, 0, 1, 1, 0]], dtype=float)

    px, py = fat_band_polygons(x, y, halfwidth, scale=(1, 1), min_halfwidth=0.1)

    polygons = _polygons(px, py)
    assert [(p[:, 0].min(), p[:, 0].max()) for p in polygons] == [(1, 3), (3, 5)]

    px, py = fat_band_polygons(x, 0 * halfwidth, 0 * halfwidth, (1, 1), 0.1)
    assert px.size == py.size == 0


def test_prepare_projected_bands_trace():
    bands_data = _bands_data([0, 1, 2], [[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]])
    projection = _projection([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    trace = prepare_projected_bands_trace(
        bands_data, projection, bands_width=2, scale=(1, 1), min_halfwidth=0.1
    )

    assert trace["label"] == "Si 3s"
    assert trace["color"] == "#ff0000"
    # Only the first band is projected
    assert np.nanmax(trace["y"]) == pytest.approx(1)
    assert np.nanmin(trace["y"]) == pytest.approx(-1)


@pytest.fixture
def helper():
    nkpoints = 50
    x = np.linspace(0, 1, nkpoints)
    bands_data = _bands_data(x, 10 * x[:, None])
    helper = BandsPdosPlotly(bands_data=bands_data)
    helper.project_bands = [_projection(np.ones((nkpoints, 1)))]
    return helper


def test_projected_bands_scale(helper):
    """The scale follows the layout, including the legend added by the projections."""
    fig = helper.bandspdosfigure

    # Default energy range (20 eV) over a 600 px high figure with 180 px of margins
    assert helper._projected_bands_scale(fig)[1] == pytest.approx(420 / 20)
    # 850 px wide, 80 px margins and a legend with one 5-character label
    right = (0.02 * (850 - 80) + 52 + 7 * 5) / 1.02
    assert helper._projected_bands_scale(fig, ["Si 3s"])[0] == pytest.approx(
        850 - 80 - right
    )


def test_projected_bands_update(helper):
    """Width changes and zooming with a new aspect ratio recompute the outlines."""
    fig = helper.bandspdosfigure
    helper.adding_projected_bands(fig, width=0.5)
    (trace,) = [trace for trace in fig.data if trace.legendgroup == "Si 3s"]

    helper.update_projected_bands_thickness(fig, width=2.0)
    x_before = np.array(trace.x)

    # Uniform zoom: the outlines in data coordinates stay correct
    fig.layout.xaxis.range = [0, 0.5]
    fig.layout.yaxis.range = [-5, 5]
    helper.update_projected_bands_on_relayout(fig, width=2.0)
    np.testing.assert_array_equal(trace.x, x_before)

    # Zooming only on the energy axis changes the perpendicular directions
    fig.layout.yaxis.range = [-1, 1]
    helper.update_projected_bands_on_relayout(fig, width=2.0)
    assert not np.array_equal(trace.x, x_before, equal_nan=True)


def test_energy_window():
    """Only segments whose fat band reaches into the window are drawn."""
    x = np.arange(5)
    y = np.array([[0.0] * 5, [1.2] * 5, [5.0] * 5])
    halfwidth = np.full((3, 5), 0.25)

    px, py = fat_band_polygons(x, y, halfwidth, (1, 1), y_window=(-1, 1))

    # The second band only reaches into the window with its thickness
    assert len(_polygons(px, py)) == 2
    assert np.nanmax(py) == pytest.approx(1.45)
    px, py = fat_band_polygons(x, y, halfwidth, (1, 1), y_window=(10, 20))
    assert px.size == 0


def test_projected_bands_pan(helper):
    """Panning beyond the computed energy window recomputes the outlines."""
    fig = helper.bandspdosfigure
    helper.adding_projected_bands(fig, width=0.5)
    (trace,) = [trace for trace in fig.data if trace.legendgroup == "Si 3s"]

    # Bands from 0 to 10 eV; the default range of -10 to 10 eV is extended to 20 eV
    assert helper._projected_bands_window_used == (-20, 20)
    fig.layout.yaxis.range = [-30, -10]
    helper.update_projected_bands_on_relayout(fig, width=0.5)
    assert helper._projected_bands_window_used == (-40, 0)
    # Only the start of the band is left, up to the first segment reaching 0 eV
    assert np.nanmax(trace.y) < 1
