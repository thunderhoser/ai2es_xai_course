"""Methods for sequential forward and backward selection (SFS and SBS)."""

import copy
import numpy
import sklearn.base
import sklearn.metrics
from matplotlib import pyplot
from ai2es_xai_course.utils import utils

MIN_PROB_FOR_XENTROPY = numpy.finfo(float).eps
MAX_PROB_FOR_XENTROPY = 1. - numpy.finfo(float).eps

FONT_SIZE = 30
FEATURE_NAME_FONT_SIZE = 18
FIG_WIDTH_INCHES = 15
FIG_HEIGHT_INCHES = 15
FIG_PADDING_AT_BOTTOM_PERCENT = 25.
DOTS_PER_INCH = 600

DEFAULT_BAR_FACE_COLOUR = numpy.array([228., 26., 28.]) / 255
DEFAULT_BAR_EDGE_COLOUR = numpy.array([0., 0., 0.]) / 255
DEFAULT_BAR_EDGE_WIDTH = 2.

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

MIN_FRACTIONAL_COST_DECREASE_SFS_DEFAULT = 0.01
MIN_FRACTIONAL_COST_DECREASE_SBS_DEFAULT = -0.01

DEFAULT_NUM_FORWARD_STEPS_FOR_SFS = 2
DEFAULT_NUM_BACKWARD_STEPS_FOR_SFS = 1
DEFAULT_NUM_FORWARD_STEPS_FOR_SBS = 1
DEFAULT_NUM_BACKWARD_STEPS_FOR_SBS = 2

SELECTED_FEATURES_KEY = 'selected_feature_names'
REMOVED_FEATURES_KEY = 'removed_feature_names'
FEATURE_NAME_KEY = 'feature_name'
VALIDATION_COST_KEY = 'validation_cost'
VALIDATION_XENTROPY_KEY = 'validation_cross_entropy'
VALIDATION_AUC_KEY = 'validation_auc'
TESTING_XENTROPY_KEY = 'testing_cross_entropy'
TESTING_AUC_KEY = 'testing_auc'
VALIDATION_COST_BY_STEP_KEY = 'validation_cost_by_step'

PERMUTATION_TYPE = 'permutation'
FORWARD_SELECTION_TYPE = 'forward'
BACKWARD_SELECTION_TYPE = 'backward'

MIN_PROBABILITY = 1e-15
MAX_PROBABILITY = 1. - MIN_PROBABILITY


def _plot_selection_results(
        used_feature_names, validation_cost_by_feature, selection_type,
        validation_cost_before_permutn=None, plot_feature_names=False,
        bar_face_colour=DEFAULT_BAR_FACE_COLOUR,
        bar_edge_colour=DEFAULT_BAR_EDGE_COLOUR,
        bar_edge_width=DEFAULT_BAR_EDGE_WIDTH):
    """Plots bar graph with results of feature-selection algorithm.

    K = number of features added, removed, or permuted

    :param used_feature_names: length-K list of feature names in order of their
        addition to the model, removal from the model, or permutation in the
        model's training data.
    :param validation_cost_by_feature: length-K list of validation costs.
        validation_cost_by_feature[k] is cost after adding, removing, or
        permuting feature_names[k].
    :param selection_type: Type of selection algorithm used.  May be "forward",
        "backward", or "permutation".
    :param validation_cost_before_permutn: Validation cost before permutation.
        If selection_type != "permutation", you can leave this as None.
    :param plot_feature_names: Boolean flag.  If True, will plot feature names
        on x-axis.  If False, will plot ordinal numbers on x-axis.
    :param bar_face_colour: Colour (in any format accepted by
        `matplotlib.colors`) for interior of bars.
    :param bar_edge_colour: Colour for edge of bars.
    :param bar_edge_width: Width for edge of bars.
    """

    num_features_used = len(used_feature_names)

    if selection_type == PERMUTATION_TYPE:
        x_coords_at_bar_edges = numpy.linspace(
            -0.5, num_features_used + 0.5, num=num_features_used + 2
        )

        y_values = numpy.concatenate((
            numpy.array([validation_cost_before_permutn]),
            validation_cost_by_feature
        ))
    else:
        x_coords_at_bar_edges = numpy.linspace(
            0.5, num_features_used + 0.5, num=num_features_used + 1
        )

        y_values = copy.deepcopy(validation_cost_by_feature)

    x_width_of_bar = x_coords_at_bar_edges[1] - x_coords_at_bar_edges[0]
    x_coords_at_bar_centers = x_coords_at_bar_edges[:-1] + x_width_of_bar / 2

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES)
    )

    figure_object.subplots_adjust(bottom=FIG_PADDING_AT_BOTTOM_PERCENT / 100)

    axes_object.bar(
        x_coords_at_bar_centers, y_values, x_width_of_bar,
        color=bar_face_colour, edgecolor=bar_edge_colour,
        linewidth=bar_edge_width
    )

    pyplot.xticks(x_coords_at_bar_centers, axes=axes_object)

    axes_object.set_xlim(
        numpy.min(x_coords_at_bar_edges), numpy.max(x_coords_at_bar_edges)
    )
    axes_object.set_ylim(0., 1.05 * numpy.max(y_values))

    if selection_type == PERMUTATION_TYPE:
        if plot_feature_names:
            axes_object.set_xlabel('Feature permuted')
        else:
            axes_object.set_xlabel('Number of features permuted')

    elif selection_type == FORWARD_SELECTION_TYPE:
        if plot_feature_names:
            axes_object.set_xlabel('Feature selected')
        else:
            axes_object.set_xlabel('Number of features selected')

    elif selection_type == BACKWARD_SELECTION_TYPE:
        if plot_feature_names:
            axes_object.set_xlabel('Feature removed')
        else:
            axes_object.set_xlabel('Number of features removed')

    if plot_feature_names:
        if selection_type == PERMUTATION_TYPE:
            pyplot.xticks(
                x_coords_at_bar_centers, [' '] + used_feature_names.tolist(),
                rotation='vertical', fontsize=FEATURE_NAME_FONT_SIZE
            )
        else:
            pyplot.xticks(
                x_coords_at_bar_centers, used_feature_names,
                rotation='vertical', fontsize=FEATURE_NAME_FONT_SIZE
            )

    axes_object.set_ylabel('Validation cost')


def _forward_selection_step(
        training_predictor_table, validation_predictor_table,
        training_target_table, validation_target_table,
        remaining_predictor_names, sklearn_model_object):
    """Runs one step of sequential forward selection.

    :param training_predictor_table: See doc for `run_forward_selection`.
    :param validation_predictor_table: Same.
    :param training_target_table: Same.
    :param validation_target_table: Same.
    :param remaining_predictor_names: 1-D list with names of predictors not yet
        selected.
    :param sklearn_model_object: See doc for `run_forward_selection`.
    :return: min_cost: Minimum cost given by adding any set of L features from
        `remaining_feature_names` to the model.
    :return: best_feature_names: length-L list of features whose addition
        resulted in `min_cost`.
    """

    selected_predictor_names = list(
        set(list(training_predictor_table)) -
        set(remaining_predictor_names)
    )

    num_remaining_predictors = len(remaining_predictor_names)
    one_minus_auc_values = numpy.full(num_remaining_predictors, numpy.nan)

    for j in range(num_remaining_predictors):
        these_predictor_names = (
            selected_predictor_names + [remaining_predictor_names[j]]
        )

        new_model_object = sklearn.base.clone(sklearn_model_object)
        new_model_object.fit(
            X=training_predictor_table[these_predictor_names].to_numpy(),
            y=training_target_table[utils.BINARIZED_TARGET_NAME].values
        )

        these_forecast_probs = new_model_object.predict_proba(
            validation_predictor_table[these_predictor_names].to_numpy()
        )[:, 1]

        one_minus_auc_values[j] = sklearn.metrics.roc_auc_score(
            validation_target_table[utils.BINARIZED_TARGET_NAME].values,
            these_forecast_probs
        )

    one_minus_auc_values = 1. - one_minus_auc_values

    min_cost = numpy.min(one_minus_auc_values)
    best_index = numpy.argmin(one_minus_auc_values)
    return min_cost, remaining_predictor_names[best_index]


def _backward_selection_step(
        training_predictor_table, validation_predictor_table,
        training_target_table, validation_target_table,
        selected_predictor_names, sklearn_model_object):
    """Runs one step of sequential backward selection.

    :param training_predictor_table: See doc for `run_backward_selection`.
    :param validation_predictor_table: Same.
    :param training_target_table: Same.
    :param validation_target_table: Same.
    :param selected_predictor_names: 1-D list with names of predictors not yet
        removed.
    :param sklearn_model_object: See doc for `run_backward_selection`.
    :return: min_cost: Minimum cost given by adding any set of L features from
        `remaining_feature_names` to the model.
    :return: best_feature_names: length-L list of features whose addition
        resulted in `min_cost`.
    """

    # TODO(thunderhoser): Clean output doc.

    num_selected_predictors = len(selected_predictor_names)
    one_minus_auc_values = numpy.full(num_selected_predictors, numpy.nan)

    for j in range(num_selected_predictors):
        these_predictor_names = set(selected_predictor_names)
        these_predictor_names.remove(selected_predictor_names[j])
        these_predictor_names = list(these_predictor_names)

        new_model_object = sklearn.base.clone(sklearn_model_object)
        new_model_object.fit(
            X=training_predictor_table[these_predictor_names].to_numpy(),
            y=training_target_table[utils.BINARIZED_TARGET_NAME].values
        )

        these_forecast_probs = new_model_object.predict_proba(
            validation_predictor_table[these_predictor_names].to_numpy()
        )[:, 1]

        one_minus_auc_values[j] = sklearn.metrics.roc_auc_score(
            validation_target_table[utils.BINARIZED_TARGET_NAME].values,
            these_forecast_probs
        )

    one_minus_auc_values = 1. - one_minus_auc_values

    min_cost = numpy.min(one_minus_auc_values)
    worst_index = numpy.argmin(one_minus_auc_values)
    return min_cost, selected_predictor_names[worst_index]


def run_forward_selection(
        training_predictor_table, validation_predictor_table,
        training_target_table, validation_target_table, sklearn_model_object,
        min_fractional_cost_decrease=0.01):
    """Runs sequential forward selection.

    :param training_predictor_table: pandas DataFrame with predictor values.
        Each row is one storm object in the training set.
    :param validation_predictor_table: Same but for validation set.
    :param training_target_table: pandas DataFrame with target values.  Each row
        is one storm object in the training set.
    :param validation_target_table: Same but for validation set.
    :param sklearn_model_object: Trained scikit-learn model.  Must implement the
        methods `fit` and `predict_proba`.
    :param min_fractional_cost_decrease: Stopping criterion.  Once the
        fractional cost decrease over one step is <
        `min_fractional_cost_decrease`, SFS will stop.  Must be in range (0, 1).
    :return: sfs_dictionary: Same as output from _evaluate_feature_selection,
        but with one additional key.
    sfs_dictionary['validation_cost_by_step']: length-f numpy array of
        validation costs.  The [i]th element is the cost with i features added.
        In other words, validation_cost_by_step[0] is the cost with 1 feature
        added; validation_cost_by_step[1] is the cost with 2 features added;
        ...; etc.
    """

    # TODO(thunderhoser): Clean output doc.

    assert min_fractional_cost_decrease > 0.
    assert min_fractional_cost_decrease < 1.

    # Initialize values.
    selected_predictor_names = []
    remaining_predictor_names = list(training_predictor_table)

    num_predictors = len(remaining_predictor_names)
    min_cost_by_num_selected = numpy.full(num_predictors + 1, numpy.nan)
    min_cost_by_num_selected[0] = numpy.inf

    while len(remaining_predictor_names) > 0:
        num_selected_features = len(selected_predictor_names)
        num_remaining_features = len(remaining_predictor_names)

        print((
            'Step {0:d} of sequential forward selection: {1:d} features '
            'selected, {2:d} remaining...'
        ).format(
            num_selected_features + 1, num_selected_features,
            num_remaining_features
        ))

        min_new_cost, this_best_predictor_name = _forward_selection_step(
            training_predictor_table=training_predictor_table,
            validation_predictor_table=validation_predictor_table,
            training_target_table=training_target_table,
            validation_target_table=validation_target_table,
            remaining_predictor_names=remaining_predictor_names,
            sklearn_model_object=sklearn_model_object
        )

        these_best_predictor_names = [this_best_predictor_name]

        print((
            'Minimum cost ({0:.4f}) given by adding features shown below '
            '(previous minimum = {1:.4f}).\n{2:s}\n'
        ).format(
            min_new_cost, min_cost_by_num_selected[num_selected_features],
            str(these_best_predictor_names)
        ))

        stopping_criterion = min_cost_by_num_selected[num_selected_features] * (
            1. - min_fractional_cost_decrease
        )

        if min_new_cost > stopping_criterion:
            break

        selected_predictor_names += these_best_predictor_names
        remaining_predictor_names = [
            s for s in remaining_predictor_names
            if s not in these_best_predictor_names
        ]

        min_cost_by_num_selected[
            (num_selected_features + 1):(len(selected_predictor_names) + 1)
        ] = min_new_cost

    num_selected_features = len(selected_predictor_names)

    return {
        SELECTED_FEATURES_KEY: selected_predictor_names,
        VALIDATION_COST_BY_STEP_KEY:
            min_cost_by_num_selected[1:(num_selected_features + 1)]
    }


def run_backward_selection(
        training_predictor_table, validation_predictor_table,
        training_target_table, validation_target_table, sklearn_model_object,
        min_fractional_cost_decrease=-0.01):
    """Runs sequential forward selection.

    :param training_predictor_table: pandas DataFrame with predictor values.
        Each row is one storm object in the training set.
    :param validation_predictor_table: Same but for validation set.
    :param training_target_table: pandas DataFrame with target values.  Each row
        is one storm object in the training set.
    :param validation_target_table: Same but for validation set.
    :param sklearn_model_object: Trained scikit-learn model.  Must implement the
        methods `fit` and `predict_proba`.
    :param min_fractional_cost_decrease: Stopping criterion.  Once the
        fractional cost decrease over one step is <
        `min_fractional_cost_decrease`, SBS will stop.  Must be in range
        (-1, 1).  If negative, cost may increase slightly without SBS stopping.
    :return: sfs_dictionary: Same as output from _evaluate_feature_selection,
        but with one additional key.
    sfs_dictionary['validation_cost_by_step']: length-f numpy array of
        validation costs.  The [i]th element is the cost with i features added.
        In other words, validation_cost_by_step[0] is the cost with 1 feature
        added; validation_cost_by_step[1] is the cost with 2 features added;
        ...; etc.
    """

    assert min_fractional_cost_decrease > -1.
    assert min_fractional_cost_decrease < 1.

    # Initialize values.
    removed_predictor_names = []
    selected_predictor_names = list(training_predictor_table)

    num_predictors = len(selected_predictor_names)
    min_cost_by_num_removed = numpy.full(num_predictors + 1, numpy.nan)
    min_cost_by_num_removed[0] = numpy.inf

    while len(selected_predictor_names) > 0:
        num_removed_features = len(removed_predictor_names)
        num_selected_features = len(selected_predictor_names)

        print((
            'Step {0:d} of sequential backward selection: {1:d} features '
            'removed, {2:d} remaining...'
        ).format(
            num_removed_features + 1, num_removed_features,
            num_selected_features
        ))

        min_new_cost, this_worst_predictor_name = _backward_selection_step(
            training_predictor_table=training_predictor_table,
            validation_predictor_table=validation_predictor_table,
            training_target_table=training_target_table,
            validation_target_table=validation_target_table,
            selected_predictor_names=selected_predictor_names,
            sklearn_model_object=sklearn_model_object
        )

        these_worst_predictor_names = [this_worst_predictor_name]

        print((
            'Minimum cost ({0:.4f}) given by removing features shown below '
            '(previous minimum = {1:.4f}).\n{2:s}\n'
        ).format(
            min_new_cost, min_cost_by_num_removed[num_removed_features],
            str(these_worst_predictor_names)
        ))

        stopping_criterion = min_cost_by_num_removed[num_removed_features] * (
            1. - min_fractional_cost_decrease
        )

        if min_new_cost > stopping_criterion:
            break

        removed_predictor_names += these_worst_predictor_names
        selected_predictor_names = [
            s for s in selected_predictor_names
            if s not in these_worst_predictor_names
        ]

        min_cost_by_num_removed[
            (num_removed_features + 1):(len(removed_predictor_names) + 1)
        ] = min_new_cost

    num_removed_features = len(removed_predictor_names)

    return {
        REMOVED_FEATURES_KEY: removed_predictor_names,
        VALIDATION_COST_BY_STEP_KEY:
            min_cost_by_num_removed[1:(num_removed_features + 1)]
    }


def plot_forward_selection_results(
        forward_selection_dict, plot_feature_names=False,
        bar_face_colour=DEFAULT_BAR_FACE_COLOUR,
        bar_edge_colour=DEFAULT_BAR_EDGE_COLOUR,
        bar_edge_width=DEFAULT_BAR_EDGE_WIDTH):
    """Plots bar graph with results of any forward-selection algorithm.

    :param forward_selection_dict: Dictionary returned by
        `sequential_forward_selection`, `sfs_with_backward_steps`, or
        `floating_sfs`.
    :param plot_feature_names: See documentation for _plot_selection_results.
    :param bar_face_colour: See doc for _plot_selection_results.
    :param bar_edge_colour: See doc for _plot_selection_results.
    :param bar_edge_width: See doc for _plot_selection_results.
    """

    _plot_selection_results(
        used_feature_names=forward_selection_dict[SELECTED_FEATURES_KEY],
        validation_cost_by_feature=
        forward_selection_dict[VALIDATION_COST_BY_STEP_KEY],
        selection_type=FORWARD_SELECTION_TYPE,
        plot_feature_names=plot_feature_names, bar_face_colour=bar_face_colour,
        bar_edge_colour=bar_edge_colour, bar_edge_width=bar_edge_width)


def plot_backward_selection_results(
        backward_selection_dict, plot_feature_names=False,
        bar_face_colour=DEFAULT_BAR_FACE_COLOUR,
        bar_edge_colour=DEFAULT_BAR_EDGE_COLOUR,
        bar_edge_width=DEFAULT_BAR_EDGE_WIDTH):
    """Plots bar graph with results of any backward-selection algorithm.

    :param backward_selection_dict: Dictionary returned by
        `sequential_backward_selection`, `sbs_with_forward_steps`, or
        `floating_sbs`.
    :param plot_feature_names: See documentation for _plot_selection_results.
    :param bar_face_colour: See doc for _plot_selection_results.
    :param bar_edge_colour: See doc for _plot_selection_results.
    :param bar_edge_width: See doc for _plot_selection_results.
    """

    _plot_selection_results(
        used_feature_names=backward_selection_dict[REMOVED_FEATURES_KEY],
        validation_cost_by_feature=
        backward_selection_dict[VALIDATION_COST_BY_STEP_KEY],
        selection_type=BACKWARD_SELECTION_TYPE,
        plot_feature_names=plot_feature_names, bar_face_colour=bar_face_colour,
        bar_edge_colour=bar_edge_colour, bar_edge_width=bar_edge_width)
