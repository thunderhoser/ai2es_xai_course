"""Methods for sequential forward and backward selection (SFS and SBS)."""

import numpy
import sklearn.base
import sklearn.metrics
from matplotlib import pyplot
from ai2es_xai_course.utils import utils

BAR_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
BAR_EDGE_WIDTH = 2
BAR_EDGE_COLOUR = numpy.full(3, 0.)
BAR_TEXT_COLOUR = numpy.full(3, 0.)
BAR_FONT_SIZE = 18

FIGURE_WIDTH_INCHES = 10
FIGURE_HEIGHT_INCHES = 10

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
    :return: max_validation_auc: Max AUC on validation data.
    :return: best_predictor_name: Name of predictor whose addition yields max
        validation AUC.
    """

    selected_predictor_names = list(
        set(list(training_predictor_table)) -
        set(remaining_predictor_names)
    )

    num_remaining_predictors = len(remaining_predictor_names)
    validation_auc_values = numpy.full(num_remaining_predictors, numpy.nan)

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

        validation_auc_values[j] = sklearn.metrics.roc_auc_score(
            validation_target_table[utils.BINARIZED_TARGET_NAME].values,
            these_forecast_probs
        )

    max_validation_auc = numpy.max(validation_auc_values)
    best_index = numpy.argmax(validation_auc_values)
    return max_validation_auc, remaining_predictor_names[best_index]


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
    :return: max_validation_auc: Max AUC on validation data.
    :return: worst_predictor_name: Name of predictor whose removal yields max
        validation AUC.
    """

    num_selected_predictors = len(selected_predictor_names)
    validation_auc_values = numpy.full(num_selected_predictors, numpy.nan)

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

        validation_auc_values[j] = sklearn.metrics.roc_auc_score(
            validation_target_table[utils.BINARIZED_TARGET_NAME].values,
            these_forecast_probs
        )

    max_validation_auc = numpy.max(validation_auc_values)
    worst_index = numpy.argmax(validation_auc_values)
    return max_validation_auc, selected_predictor_names[worst_index]


def run_forward_selection(
        training_predictor_table, validation_predictor_table,
        training_target_table, validation_target_table, sklearn_model_object,
        min_fractional_cost_decrease=0.01):
    """Runs sequential forward selection.

    P = number of predictors selected

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
    :return: selected_predictor_names: length-P list of predictor names.
    :return: validation_auc_values: length-P numpy array of AUC values on
        validation data.
    """

    assert min_fractional_cost_decrease > 0.
    assert min_fractional_cost_decrease < 1.

    selected_predictor_names = []
    max_auc_values = [-numpy.inf]
    remaining_predictor_names = list(training_predictor_table)

    while len(remaining_predictor_names) > 0:
        num_selected_predictors = len(selected_predictor_names)
        num_remaining_predictors = len(remaining_predictor_names)

        print((
            'Step {0:d} of forward selection: {1:d} predictors selected, '
            '{2:d} remaining...'
        ).format(
            num_selected_predictors + 1, num_selected_predictors,
            num_remaining_predictors
        ))

        new_max_auc, best_predictor_name = _forward_selection_step(
            training_predictor_table=training_predictor_table,
            validation_predictor_table=validation_predictor_table,
            training_target_table=training_target_table,
            validation_target_table=validation_target_table,
            remaining_predictor_names=remaining_predictor_names,
            sklearn_model_object=sklearn_model_object
        )

        print((
            'Max validation AUC ({0:.4f}) given by adding {1:s} '
            '(previous max AUC = {2:.4f})'
        ).format(
            new_max_auc, best_predictor_name, max_auc_values[-1]
        ))

        stopping_criterion = max_auc_values[-1] * (
            1. + min_fractional_cost_decrease
        )

        if new_max_auc < stopping_criterion:
            break

        selected_predictor_names.append(best_predictor_name)
        max_auc_values.append(new_max_auc)

        remaining_predictor_names = list(
            set(list(training_predictor_table)) -
            set(selected_predictor_names)
        )

    return selected_predictor_names, numpy.array(max_auc_values[1:])


def run_backward_selection(
        training_predictor_table, validation_predictor_table,
        training_target_table, validation_target_table, sklearn_model_object,
        min_fractional_cost_decrease=-0.01):
    """Runs sequential backward selection.

    P = number of predictors removed

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
    :return: removed_predictor_names: length-P list of predictor names.
    :return: validation_auc_values: length-P numpy array of AUC values on
        validation data.
    """

    assert min_fractional_cost_decrease > -1.
    assert min_fractional_cost_decrease < 1.

    removed_predictor_names = []
    max_auc_values = [-numpy.inf]
    selected_predictor_names = list(training_predictor_table)

    while len(selected_predictor_names) > 0:
        num_removed_predictors = len(removed_predictor_names)
        num_selected_predictors = len(selected_predictor_names)

        print((
            'Step {0:d} of backward selection: {1:d} predictors removed, '
            '{2:d} remaining...'
        ).format(
            num_removed_predictors + 1, num_removed_predictors,
            num_selected_predictors
        ))

        new_max_auc, worst_predictor_name = _backward_selection_step(
            training_predictor_table=training_predictor_table,
            validation_predictor_table=validation_predictor_table,
            training_target_table=training_target_table,
            validation_target_table=validation_target_table,
            selected_predictor_names=selected_predictor_names,
            sklearn_model_object=sklearn_model_object
        )

        print((
            'Max validation AUC ({0:.4f}) given by removing {1:s} '
            '(previous max AUC = {2:.4f})'
        ).format(
            new_max_auc, worst_predictor_name, max_auc_values[-1]
        ))

        stopping_criterion = max_auc_values[-1] * (
            1. + min_fractional_cost_decrease
        )

        if new_max_auc < stopping_criterion:
            break

        removed_predictor_names.append(worst_predictor_name)
        max_auc_values.append(new_max_auc)

        selected_predictor_names = list(
            set(list(training_predictor_table)) -
            set(removed_predictor_names)
        )

    return removed_predictor_names, numpy.array(max_auc_values[1:])


def plot_results(
        predictor_names, validation_auc_values, is_forward_test,
        axes_object=None, num_predictors_to_plot=None):
    """Plots results of SFS or SBS.

    P = number of predictors selected (or removed if `is_forward_test == False`)

    :param predictor_names: length-P list of predictor names.
    :param validation_auc_values: length-P numpy array of areas under ROC curve
        (AUC) on validation data.
    :param is_forward_test: Boolean flag.
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If None, will create new
        axes.
    :param num_predictors_to_plot: Number of predictors to plot.  Will plot only
        the first K in the list, where K = `num_predictors_to_plot`.  If None,
        will plot all predictors in the list.
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    if num_predictors_to_plot is None:
        num_predictors_to_plot = len(predictor_names)

    num_predictors_to_plot = max([num_predictors_to_plot, 2])
    num_predictors_to_plot = min([
        num_predictors_to_plot, len(predictor_names)
    ])

    y_coords = numpy.linspace(
        0, num_predictors_to_plot - 1, num=num_predictors_to_plot, dtype=float
    )

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    axes_object.barh(
        y_coords, validation_auc_values[:num_predictors_to_plot],
        color=BAR_FACE_COLOUR, edgecolor=BAR_EDGE_COLOUR,
        linewidth=BAR_EDGE_WIDTH
    )

    axes_object.set_yticks([], [])
    axes_object.set_xlabel('Validation AUC')

    if is_forward_test:
        axes_object.set_ylabel('Predictor added')
    else:
        axes_object.set_ylabel('Predictor removed')

    axes_object.set_ylim(
        numpy.min(y_coords) - 0.75, numpy.max(y_coords) + 0.75
    )
    axes_object.set_xlim(
        0, numpy.max(validation_auc_values[:num_predictors_to_plot])
    )

    for j in range(num_predictors_to_plot):
        axes_object.text(
            0., y_coords[j], '      ' + predictor_names[j],
            color=BAR_TEXT_COLOUR, fontsize=BAR_FONT_SIZE,
            horizontalalignment='left', verticalalignment='center'
        )

    return axes_object
