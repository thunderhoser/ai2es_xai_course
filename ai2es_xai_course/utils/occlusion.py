"""Helper methods for occlusion."""

import numpy
from ai2es_xai_course.utils import utils
from ai2es_xai_course.utils import cnn

DEFAULT_LINE_WIDTH = 2.


def _get_grid_points(x_min, x_spacing, num_columns, y_min, y_spacing, num_rows):
    """Returns grid points in regular x-y grid.

    M = number of rows in grid
    N = number of columns in grid

    :param x_min: Minimum x-coordinate over all grid points.
    :param x_spacing: Spacing between adjacent grid points in x-direction.
    :param num_columns: N in the above definition.
    :param y_min: Minimum y-coordinate over all grid points.
    :param y_spacing: Spacing between adjacent grid points in y-direction.
    :param num_rows: M in the above definition.
    :return: x_coords: length-N numpy array with x-coordinates at grid points.
    :return: y_coords: length-M numpy array with y-coordinates at grid points.
    """

    # TODO(thunderhoser): Put this in utils.py.

    x_max = x_min + (num_columns - 1) * x_spacing
    y_max = y_min + (num_rows - 1) * y_spacing

    x_coords = numpy.linspace(x_min, x_max, num=num_columns)
    y_coords = numpy.linspace(y_min, y_max, num=num_rows)

    return x_coords, y_coords


def get_occlusion_maps(
        model_object, predictor_matrix, half_window_size_px=1, fill_value=0.):
    """Computes occlusion map for each example for the positive class.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (predictor variables)

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictors.
    :param half_window_size_px: Half-size of occlusion window (pixels).  If
        half-size is P, the full window will (2 * P + 1) rows by (2 * P + 1)
        columns.
    :param fill_value: Fill value.  Inside the occlusion window, all channels
        will be assigned this value, to simulate missing data.
    :return: occlusion_prob_matrix: E-by-M-by-N numpy array of predicted
        probabilities after occlusion.
    :return: original_probs: length-E numpy array of predicted probabilities
        before occlusion.
    """

    half_window_size_px = int(numpy.round(half_window_size_px))
    assert half_window_size_px >= 0

    assert not numpy.any(numpy.isnan(predictor_matrix))
    assert len(predictor_matrix.shape) == 4

    num_grid_rows = predictor_matrix.shape[1]
    num_grid_columns = predictor_matrix.shape[2]
    occlusion_prob_matrix = numpy.full(predictor_matrix.shape[:-1], numpy.nan)

    for i in range(num_grid_rows):
        print('Occluding windows centered on row {0:d}...'.format(i + 1))

        for j in range(num_grid_columns):
            first_row = max([i - half_window_size_px, 0])
            last_row = min([i + half_window_size_px + 1, num_grid_rows])
            first_column = max([j - half_window_size_px, 0])
            last_column = min([j + half_window_size_px + 1, num_grid_columns])

            new_predictor_matrix = predictor_matrix + 0.
            new_predictor_matrix[
                :, first_row:last_row, first_column:last_column, :
            ] = fill_value

            occlusion_prob_matrix[:, i, j] = cnn.apply_model(
                model_object=model_object,
                predictor_matrix=new_predictor_matrix, verbose=False
            )

    original_probs = cnn.apply_model(
        model_object=model_object,
        predictor_matrix=predictor_matrix, verbose=False
    )

    return occlusion_prob_matrix, original_probs


def smooth_occlusion_maps(occlusion_prob_matrix, smoothing_radius_grid_cells):
    """Smooths occlusion maps via Gaussian filter.

    :param occlusion_prob_matrix: See output doc for `get_occlusion_maps`.
    :param smoothing_radius_grid_cells: e-folding radius (number of grid cells).
    :return: occlusion_prob_matrix: Smoothed version of input.
    """

    num_examples = occlusion_prob_matrix.shape[0]

    for i in range(num_examples):
        occlusion_prob_matrix[i, ...] = utils.apply_gaussian_filter(
            input_matrix=occlusion_prob_matrix[i, ...],
            e_folding_radius_grid_cells=smoothing_radius_grid_cells
        )

    return occlusion_prob_matrix


def normalize_occlusion_maps(occlusion_prob_matrix, original_probs):
    """Normalizes occlusion maps (scales to range -inf...1).

    :param occlusion_prob_matrix: See output doc for `get_occlusion_maps`.
    :param original_probs: Same.
    :return: normalized_occlusion_matrix: numpy array with same shape as input,
        except that each value is now a normalized *decrease* in probability.
        A value of 1 means that probability decreases all the way zero; a value
        of 0 means that probability does not decrease at all; a value of -1
        means that probability doubles; ...; etc.
    """

    assert not numpy.any(numpy.isnan(occlusion_prob_matrix))
    assert len(occlusion_prob_matrix.shape) == 3
    assert numpy.all(occlusion_prob_matrix >= 0.)
    assert numpy.all(occlusion_prob_matrix <= 1.)

    num_examples = occlusion_prob_matrix.shape[0]
    assert not numpy.any(numpy.isnan(original_probs))
    assert len(original_probs) == num_examples

    normalized_occlusion_matrix = numpy.full(
        occlusion_prob_matrix.shape, numpy.nan
    )

    original_probs_with_nan = original_probs + 0.
    original_probs_with_nan[original_probs_with_nan == 0] = numpy.nan

    for i in range(num_examples):
        normalized_occlusion_matrix[i, ...] = (
            (original_probs_with_nan[i] - occlusion_prob_matrix[i, ...]) /
            original_probs_with_nan[i]
        )

    normalized_occlusion_matrix[numpy.isnan(normalized_occlusion_matrix)] = 0.
    return normalized_occlusion_matrix


def plot_normalized_occlusion_map(
        normalized_occlusion_matrix_2d, axes_object, colour_map_object,
        max_contour_value, contour_interval, line_width=DEFAULT_LINE_WIDTH):
    """Plots 2-D normalized occlusion map with line contours.

    :param normalized_occlusion_matrix_2d: See output doc for
        `normalize_occlusion_maps`.
    :param axes_object: See input doc for `plot_occlusion_map`.
    :param colour_map_object: Same.
    :param max_contour_value: Same.
    :param contour_interval: Same.
    :param line_width: Same.
    """

    # Check input args.
    assert max_contour_value >= 0.
    max_contour_value = max([max_contour_value, 1e-6])

    assert contour_interval >= 0.
    contour_interval = max([contour_interval, 1e-7])
    assert contour_interval < max_contour_value

    assert not numpy.any(numpy.isnan(normalized_occlusion_matrix_2d))
    assert len(normalized_occlusion_matrix_2d.shape) == 2
    assert numpy.all(normalized_occlusion_matrix_2d <= 1.)

    half_num_contours = int(numpy.round(
        1 + max_contour_value / contour_interval
    ))

    # Find grid coordinates.
    num_grid_rows = normalized_occlusion_matrix_2d.shape[0]
    num_grid_columns = normalized_occlusion_matrix_2d.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    x_coords, y_coords = _get_grid_points(
        x_min=x_coord_spacing / 2, y_min=y_coord_spacing / 2,
        x_spacing=x_coord_spacing, y_spacing=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords, y_coords)

    # Plot positive contours.
    positive_contour_values = numpy.linspace(
        0., max_contour_value, num=half_num_contours
    )

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, normalized_occlusion_matrix_2d,
        positive_contour_values, cmap=colour_map_object,
        vmin=numpy.min(positive_contour_values),
        vmax=numpy.max(positive_contour_values),
        linewidths=line_width, linestyles='solid', zorder=1e6,
        transform=axes_object.transAxes
    )

    # Plot negative contours.
    negative_contour_values = positive_contour_values[1:]

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, -normalized_occlusion_matrix_2d,
        negative_contour_values, cmap=colour_map_object,
        vmin=numpy.min(negative_contour_values),
        vmax=numpy.max(negative_contour_values),
        linewidths=line_width, linestyles='dashed', zorder=1e6,
        transform=axes_object.transAxes
    )


def plot_occlusion_map(
        occlusion_prob_matrix_2d, axes_object, colour_map_object,
        min_contour_value, max_contour_value, contour_interval,
        line_width=DEFAULT_LINE_WIDTH):
    """Plots 2-D occlusion map with line contours.

    M = number of rows in grid
    N = number of columns in grid

    :param occlusion_prob_matrix_2d: M-by-N numpy array of probabilities after
        occlusion.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param min_contour_value: Minimum contour value.
    :param max_contour_value: Max contour value.
    :param contour_interval: Interval between successive contours.
    :param line_width: Line width for contours.
    """

    # Check input args.
    assert min_contour_value >= 0.
    assert max_contour_value >= 0.
    min_contour_value = max([min_contour_value, 1e-6])
    max_contour_value = max([max_contour_value, 1e-5])

    assert contour_interval >= 0.
    contour_interval = max([contour_interval, 1e-7])

    assert not numpy.any(numpy.isnan(occlusion_prob_matrix_2d))
    assert len(occlusion_prob_matrix_2d.shape) == 2
    assert numpy.all(occlusion_prob_matrix_2d >= 0.)
    assert numpy.all(occlusion_prob_matrix_2d <= 1.)

    assert contour_interval < max_contour_value

    num_contours = int(numpy.round(
        1 + (max_contour_value - min_contour_value) / contour_interval
    ))

    # Find grid coordinates.
    num_grid_rows = occlusion_prob_matrix_2d.shape[0]
    num_grid_columns = occlusion_prob_matrix_2d.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    x_coords, y_coords = _get_grid_points(
        x_min=x_coord_spacing / 2, y_min=y_coord_spacing / 2,
        x_spacing=x_coord_spacing, y_spacing=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords, y_coords)

    contour_values = numpy.linspace(
        min_contour_value, max_contour_value, num=num_contours
    )

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, occlusion_prob_matrix_2d,
        contour_values, cmap=colour_map_object,
        vmin=min_contour_value, vmax=max_contour_value,
        linewidths=line_width, linestyles='solid', zorder=1e6,
        transform=axes_object.transAxes
    )
