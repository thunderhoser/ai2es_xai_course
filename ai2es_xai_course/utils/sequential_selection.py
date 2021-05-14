"""Methods for sequential forward and backward selection (SFS and SBS)."""

import copy
from itertools import combinations
import numpy
import keras.utils
import sklearn.base
import sklearn.metrics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils

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


def _check_sequential_selection_inputs(
        training_table, validation_table, feature_names, target_name,
        num_features_to_add_per_step=1, num_features_to_remove_per_step=1,
        testing_table=None):
    """Checks inputs for sequential forward or backward selection.

    :param training_table: pandas DataFrame, where each row is one training
        example.
    :param validation_table: pandas DataFrame, where each row is one validation
        example.
    :param feature_names: length-F list with names of features (predictor
        variables).  Each feature must be a column in training_table,
        validation_table, and testing_table.
    :param target_name: Name of target variable (predictand).  Must be a column
        in training_table, validation_table, and testing_table.
    :param num_features_to_add_per_step: Number of features to add at each
        forward step.
    :param num_features_to_remove_per_step: Number of features to remove at each
        backward step.
    :param testing_table: pandas DataFrame, where each row is one testing
        example.
    """

    error_checking.assert_is_string_list(feature_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(feature_names), num_dimensions=1)

    error_checking.assert_is_string(target_name)
    variable_names = feature_names + [target_name]
    error_checking.assert_columns_in_dataframe(training_table, variable_names)
    error_checking.assert_columns_in_dataframe(validation_table, variable_names)
    if testing_table is not None:
        error_checking.assert_columns_in_dataframe(
            testing_table, variable_names)

    num_features = len(feature_names)
    error_checking.assert_is_integer(num_features_to_add_per_step)
    error_checking.assert_is_geq(num_features_to_add_per_step, 1)
    error_checking.assert_is_less_than(
        num_features_to_add_per_step, num_features)

    error_checking.assert_is_integer(num_features_to_remove_per_step)
    error_checking.assert_is_geq(num_features_to_remove_per_step, 1)
    error_checking.assert_is_less_than(
        num_features_to_remove_per_step, num_features)

    # Ensure that label is binary.
    error_checking.assert_is_integer_numpy_array(
        training_table[target_name].values)
    error_checking.assert_is_geq_numpy_array(
        training_table[target_name].values, 0)
    error_checking.assert_is_leq_numpy_array(
        training_table[target_name].values, 1)


def _forward_selection_step(
        training_table, validation_table, selected_feature_names,
        remaining_feature_names, target_name, estimator_object, cost_function,
        num_features_to_add=1):
    """Performs one forward selection step (i.e., adds features to the model).

    The best set of L features is added to the model, where L >= 1.

    Each member of `selected_feature_names` and `remaining_feature_names`, as
    well as `target_name`, must be a column in both `training_table` and
    `validation_table`.

    :param training_table: pandas DataFrame, where each row is one training
        example.
    :param validation_table: pandas DataFrame, where each row is one validation
        example.
    :param selected_feature_names: 1-D list with names of selected features
        (those already in the model).
    :param remaining_feature_names: 1-D list with names of remaining features
        (those which may be added to the model).
    :param target_name: Name of target variable (predictand).
    :param estimator_object: Instance of scikit-learn estimator.  Must implement
        the methods `fit` and `predict_proba`.
    :param cost_function: Cost function to be minimized.  Should have the
        following format.  E = number of examples, and K = number of classes.
    Input: class_probability_matrix: E-by-K numpy array of predicted
        probabilities, where class_probability_matrix[i, k] = probability that
        [i]th example belongs to [k]th class.
    Input: observed_values: length-E numpy array of observed values (integer
        class labels).
    Output: cost: Scalar value.

    :param num_features_to_add: Number of features to add (L in the above
        discussion).
    :return: min_cost: Minimum cost given by adding any set of L features from
        `remaining_feature_names` to the model.
    :return: best_feature_names: length-L list of features whose addition
        resulted in `min_cost`.
    """

    combination_object = combinations(
        remaining_feature_names, num_features_to_add)
    list_of_remaining_feature_combos = []
    for this_tuple in list(combination_object):
        list_of_remaining_feature_combos.append(list(this_tuple))

    num_remaining_feature_combos = len(list_of_remaining_feature_combos)
    cost_by_feature_combo = numpy.full(num_remaining_feature_combos, numpy.nan)

    for j in range(num_remaining_feature_combos):
        these_feature_names = (
            selected_feature_names + list_of_remaining_feature_combos[j])

        new_estimator_object = sklearn.base.clone(estimator_object)
        new_estimator_object.fit(
            training_table.as_matrix(columns=these_feature_names),
            training_table[target_name].values)

        this_probability_matrix = new_estimator_object.predict_proba(
            validation_table.as_matrix(columns=these_feature_names))
        cost_by_feature_combo[j] = cost_function(
            this_probability_matrix, validation_table[target_name].values)

    min_cost = numpy.min(cost_by_feature_combo)
    best_index = numpy.argmin(cost_by_feature_combo)
    return min_cost, list_of_remaining_feature_combos[best_index]


def _backward_selection_step(
        training_table, validation_table, selected_feature_names, target_name,
        estimator_object, cost_function, num_features_to_remove=1):
    """Performs one backward selection step (removes features from the model).

    The worst set of R features is removed from the model, where R >= 1.

    Each member of `selected_feature_names`, as well as `target_name`, must be a
    column in both `training_table` and `validation_table`.

    :param training_table: pandas DataFrame, where each row is one training
        example.
    :param validation_table: pandas DataFrame, where each row is one validation
        example.
    :param selected_feature_names: 1-D list with names of selected features
        (each of which may be removed from the model).
    :param target_name: Name of target variable (predictand).
    :param estimator_object: Instance of scikit-learn estimator.  Must implement
        the methods `fit` and `predict_proba`.
    :param cost_function: See doc for `_forward_selection_step`.
    :param num_features_to_remove: Number of features to remove (R in the above
        discussion).
    :return: min_cost: Minimum cost given by removing any set of R features in
        `selected_feature_names` from the model.
    :return: best_feature_names: length-R list of features whose removal
        resulted in `min_cost`.
    """

    combination_object = combinations(
        selected_feature_names, num_features_to_remove)
    list_of_selected_feature_combos = []
    for this_tuple in list(combination_object):
        list_of_selected_feature_combos.append(list(this_tuple))

    num_selected_feature_combos = len(list_of_selected_feature_combos)
    cost_by_feature_combo = numpy.full(num_selected_feature_combos, numpy.nan)

    for j in range(num_selected_feature_combos):
        these_feature_names = set(selected_feature_names)
        for this_name in list_of_selected_feature_combos[j]:
            these_feature_names.remove(this_name)
        these_feature_names = list(these_feature_names)

        new_estimator_object = sklearn.base.clone(estimator_object)
        new_estimator_object.fit(
            training_table.as_matrix(columns=these_feature_names),
            training_table[target_name].values)

        this_probability_matrix = new_estimator_object.predict_proba(
            validation_table.as_matrix(columns=these_feature_names))
        cost_by_feature_combo[j] = cost_function(
            this_probability_matrix, validation_table[target_name].values)

    min_cost = numpy.min(cost_by_feature_combo)
    worst_index = numpy.argmin(cost_by_feature_combo)
    return min_cost, list_of_selected_feature_combos[worst_index]


def _evaluate_feature_selection(
        training_table, validation_table, testing_table, estimator_object,
        selected_feature_names, target_name):
    """Evaluates feature selection.

    Specifically, this method computes 4 performance metrics:

    - validation cross-entropy
    - validation AUC (area under ROC curve)
    - testing cross-entropy
    - testing AUC

    :param training_table: pandas DataFrame, where each row is one training
        example.
    :param validation_table: pandas DataFrame, where each row is one validation
        example.
    :param testing_table: pandas DataFrame, where each row is one testing
        example.
    :param estimator_object: Instance of scikit-learn estimator.  Must implement
        the methods `fit` and `predict_proba`.
    :param selected_feature_names: 1-D list with names of selected features.
        Each one must be a column in training_table, validation_table, and
        testing_table.
    :param target_name: Name of target variable (predictand).  Must be a column
        in training_table, validation_table, and testing_table.
    :return: feature_selection_dict: Dictionary with the following keys.
    feature_selection_dict['selected_feature_names']: 1-D list with names of
        selected features.
    feature_selection_dict['validation_cross_entropy']: Cross-entropy on
        validation data.
    feature_selection_dict['validation_auc']: Area under ROC curve on validation
        data.
    feature_selection_dict['testing_cross_entropy']: Cross-entropy on testing
        data.
    feature_selection_dict['testing_auc']: Area under ROC curve on testing data.
    """

    new_estimator_object = sklearn.base.clone(estimator_object)
    new_estimator_object.fit(
        training_table.as_matrix(columns=selected_feature_names),
        training_table[target_name].values)

    forecast_probs_for_validation = new_estimator_object.predict_proba(
        validation_table.as_matrix(columns=selected_feature_names))[:, 1]
    validation_cross_entropy = model_eval.get_cross_entropy(
        forecast_probs_for_validation,
        validation_table[target_name].values)
    validation_auc = sklearn.metrics.roc_auc_score(
        validation_table[target_name].values, forecast_probs_for_validation)

    forecast_probs_for_testing = new_estimator_object.predict_proba(
        testing_table.as_matrix(columns=selected_feature_names))[:, 1]
    testing_cross_entropy = model_eval.get_cross_entropy(
        forecast_probs_for_testing, testing_table[target_name].values)
    testing_auc = sklearn.metrics.roc_auc_score(
        testing_table[target_name].values, forecast_probs_for_testing)

    return {SELECTED_FEATURES_KEY: selected_feature_names,
            VALIDATION_XENTROPY_KEY: validation_cross_entropy,
            VALIDATION_AUC_KEY: validation_auc,
            TESTING_XENTROPY_KEY: testing_cross_entropy,
            TESTING_AUC_KEY: testing_auc}


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
        color=plotting_utils.colour_from_numpy_to_tuple(bar_face_colour),
        edgecolor=plotting_utils.colour_from_numpy_to_tuple(bar_edge_colour),
        linewidth=bar_edge_width)

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


def _cross_entropy_function(class_probability_matrix, observed_values):
    """Cross-entropy cost function.

    This function works for binary or multi-class classification.

    E = number of examples
    K = number of classes

    :param class_probability_matrix: E-by-K numpy array of predicted
        probabilities, where class_probability_matrix[i, k] = probability that
        [i]th example belongs to [k]th class.
    :param observed_values: length-E numpy array of observed values (integer
        class labels).
    :return: cross_entropy: Scalar.
    """

    num_examples = class_probability_matrix.shape[0]
    num_classes = class_probability_matrix.shape[1]

    class_probability_matrix[
        class_probability_matrix < MIN_PROBABILITY
        ] = MIN_PROBABILITY
    class_probability_matrix[
        class_probability_matrix > MAX_PROBABILITY
        ] = MAX_PROBABILITY

    target_matrix = keras.utils.to_categorical(
        observed_values, num_classes
    ).astype(int)

    return -1 * numpy.sum(
        target_matrix * numpy.log2(class_probability_matrix)
    ) / num_examples


def sequential_forward_selection(
        training_table, validation_table, testing_table, feature_names,
        target_name, estimator_object, cost_function=_cross_entropy_function,
        num_features_to_add_per_step=1, min_fractional_cost_decrease=
        MIN_FRACTIONAL_COST_DECREASE_SFS_DEFAULT):
    """Runs the SFS (sequential forward selection) algorithm.

    SFS is defined in Chapter 9 of Webb (2003).

    f = number of features selected

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param testing_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: Instance of scikit-learn estimator.  Must implement
        the methods `fit` and `predict_proba`.
    :param cost_function: See doc for `_forward_selection_step`.
    :param num_features_to_add_per_step: Number of features to add at each step.
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

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name,
        num_features_to_add_per_step=num_features_to_add_per_step)

    error_checking.assert_is_greater(min_fractional_cost_decrease, 0.)
    error_checking.assert_is_less_than(min_fractional_cost_decrease, 1.)

    # Initialize values.
    selected_feature_names = []
    remaining_feature_names = copy.deepcopy(feature_names)

    num_features = len(feature_names)
    min_cost_by_num_selected = numpy.full(num_features + 1, numpy.nan)
    min_cost_by_num_selected[0] = numpy.inf

    while len(remaining_feature_names) >= num_features_to_add_per_step:
        num_selected_features = len(selected_feature_names)
        num_remaining_features = len(remaining_feature_names)

        print((
            'Step {0:d} of sequential forward selection: {1:d} features '
            'selected, {2:d} remaining...'
        ).format(
            num_selected_features + 1, num_selected_features,
            num_remaining_features
        ))

        min_new_cost, these_best_feature_names = _forward_selection_step(
            training_table=training_table, validation_table=validation_table,
            selected_feature_names=selected_feature_names,
            remaining_feature_names=remaining_feature_names,
            target_name=target_name, estimator_object=estimator_object,
            cost_function=cost_function,
            num_features_to_add=num_features_to_add_per_step)

        print((
            'Minimum cost ({0:.4f}) given by adding features shown below '
            '(previous minimum = {1:.4f}).\n{2:s}\n'
        ).format(
            min_new_cost, min_cost_by_num_selected[num_selected_features],
            str(these_best_feature_names)
        ))

        stopping_criterion = min_cost_by_num_selected[num_selected_features] * (
            1. - min_fractional_cost_decrease
        )

        if min_new_cost > stopping_criterion:
            break

        selected_feature_names += these_best_feature_names
        remaining_feature_names = [
            s for s in remaining_feature_names
            if s not in these_best_feature_names
        ]

        min_cost_by_num_selected[
            (num_selected_features + 1):(len(selected_feature_names) + 1)
        ] = min_new_cost

    sfs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_selected_features = len(selected_feature_names)

    sfs_dictionary.update({
        VALIDATION_COST_BY_STEP_KEY:
            min_cost_by_num_selected[1:(num_selected_features + 1)]
    })

    return sfs_dictionary


def sequential_backward_selection(
        training_table, validation_table, testing_table, feature_names,
        target_name, estimator_object, cost_function=_cross_entropy_function,
        num_features_to_remove_per_step=1, min_fractional_cost_decrease=
        MIN_FRACTIONAL_COST_DECREASE_SBS_DEFAULT):
    """Runs the SBS (sequential backward selection) algorithm.

    SBS is defined in Chapter 9 of Webb (2003).

    f = number of features selected

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param testing_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: See doc for sequential_forward_selection.
    :param cost_function: See doc for `_forward_selection_step`.
    :param num_features_to_remove_per_step: Number of features to remove at each
        step.
    :param min_fractional_cost_decrease: Stopping criterion.  Once the
        fractional cost decrease over one step is <
        `min_fractional_cost_decrease`, SBS will stop.  Must be in range
        (-1, 1).  If negative, cost may increase slightly without SBS stopping.
    :return: sbs_dictionary: Same as output from _evaluate_feature_selection,
        but with two additional keys.
    sbs_dictionary['removed_feature_names']: length-f list with names of
        features removed (in order of their removal).
    sbs_dictionary['validation_cost_by_step']: length-f numpy array of
        validation costs.  The [i]th element is the cost with i features
        removed.  In other words, validation_cost_by_step[0] is the cost with 1
        feature removed; validation_cost_by_step[1] is the cost with 2 features
        removed; ...; etc.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name,
        num_features_to_remove_per_step=num_features_to_remove_per_step)

    error_checking.assert_is_greater(min_fractional_cost_decrease, -1.)
    error_checking.assert_is_less_than(min_fractional_cost_decrease, 1.)

    # Initialize values.
    removed_feature_names = []
    selected_feature_names = copy.deepcopy(feature_names)

    num_features = len(feature_names)
    min_cost_by_num_removed = numpy.full(num_features + 1, numpy.nan)
    min_cost_by_num_removed[0] = numpy.inf

    while len(selected_feature_names) >= num_features_to_remove_per_step:
        num_removed_features = len(removed_feature_names)
        num_selected_features = len(selected_feature_names)

        print((
            'Step {0:d} of sequential backward selection: {1:d} features '
            'removed, {2:d} remaining...'
        ).format(
            num_removed_features + 1, num_removed_features,
            num_selected_features
        ))

        min_new_cost, these_worst_feature_names = _backward_selection_step(
            training_table=training_table, validation_table=validation_table,
            selected_feature_names=selected_feature_names,
            target_name=target_name, estimator_object=estimator_object,
            cost_function=cost_function,
            num_features_to_remove=num_features_to_remove_per_step)

        print((
            'Minimum cost ({0:.4f}) given by removing features shown below '
            '(previous minimum = {1:.4f}).\n{2:s}\n'
        ).format(
            min_new_cost, min_cost_by_num_removed[num_removed_features],
            str(these_worst_feature_names)
        ))

        stopping_criterion = min_cost_by_num_removed[num_removed_features] * (
            1. - min_fractional_cost_decrease
        )

        if min_new_cost > stopping_criterion:
            break

        removed_feature_names += these_worst_feature_names
        selected_feature_names = [
            s for s in selected_feature_names
            if s not in these_worst_feature_names
        ]

        min_cost_by_num_removed[
            (num_removed_features + 1):(len(removed_feature_names) + 1)
        ] = min_new_cost

    sbs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_removed_features = len(removed_feature_names)

    sbs_dictionary.update({
        REMOVED_FEATURES_KEY: removed_feature_names,
        VALIDATION_COST_BY_STEP_KEY:
            min_cost_by_num_removed[1:(num_removed_features + 1)]
    })

    return sbs_dictionary


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
