"""Helper methods for integrated gradients."""

import numpy
import tensorflow
from keras import backend as K


def _interp_predictors_one_example(predictor_matrix, num_steps):
    """For one example, interpolates each predictor between 0 and actual value.

    S = number of interpolation steps
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels

    :param predictor_matrix: M-by-N-by-C numpy array of actual predictor values.
    :param num_steps: Number of interpolation steps.
    :return: predictor_matrix_interp: S-by-M-by-N-by-C numpy array of
        interpolated predictor values.
    """

    predictor_matrix_actual = numpy.expand_dims(predictor_matrix, axis=0)
    predictor_matrix_baseline = numpy.full(predictor_matrix_actual.shape, 0.)
    difference_matrix = predictor_matrix_actual - predictor_matrix_baseline

    fractional_increments = numpy.linspace(0, 1, num=num_steps + 1, dtype=float)
    fractional_increments = numpy.expand_dims(fractional_increments, axis=-1)
    fractional_increments = numpy.expand_dims(fractional_increments, axis=-1)
    fractional_increments = numpy.expand_dims(fractional_increments, axis=-1)

    return predictor_matrix_baseline + fractional_increments * difference_matrix


def _compute_gradients(model_object, predictor_matrix_interp, target_class):
    """Computes gradient of class probability with respect to each predictor.

    S = number of interpolation steps
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels

    :param model_object: Trained neural net (instance of `keras.models.Model`
        or `keras.models.Sequential`).
    :param predictor_matrix_interp: S-by-M-by-N-by-C numpy array of
        interpolated predictor values.
    :param target_class: Array index for class we care about.
    :return: gradient_matrix: S-by-M-by-N-by-C numpy array of gradients.
    """

    num_output_neurons = (
        model_object.layers[-1].output.get_shape().as_list()[-1]
    )
    predictor_tensor_interp = tensorflow.constant(
        predictor_matrix_interp, dtype=tensorflow.float64
    )

    with tensorflow.GradientTape() as tape_object:
        tape_object.watch(predictor_tensor_interp)
        probability_array = model_object.predict(
            predictor_tensor_interp, batch_size=predictor_matrix_interp.shape[0]
        )

        if num_output_neurons == 1:
            probabilities = probability_array
        else:
            probabilities = probability_array[:, target_class]

    gradient_tensor = tape_object.gradient(
        probabilities, predictor_tensor_interp
    )
    return K.eval(gradient_tensor)


def _accumulate_gradients(predictor_matrix_actual, gradient_matrix):
    """For each predictor, accumulates gradients over interpolation path.

    This method uses the Riemann trapezoidal approximation.

    S = number of interpolation steps
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels

    :param predictor_matrix_actual: M-by-N-by-C numpy array of actual predictor
        values.
    :param gradient_matrix: S-by-M-by-N-by-C numpy array of gradients.
    :return: accum_gradient_matrix: M-by-N-by-C numpy array of accumulated
        gradients.
    """

    accum_gradient_matrix = numpy.mean(
        0.5 * (gradient_matrix[:-1, ...] + gradient_matrix[1:, ...]),
        axis=0
    )
    predictor_matrix_baseline = numpy.full(predictor_matrix_actual.shape, 0.)

    return (
        (predictor_matrix_actual - predictor_matrix_baseline) *
        accum_gradient_matrix
    )


def run_integrated_gradients(
        model_object, predictor_matrix, target_class, num_interp_steps=50):
    """Runs the integrated-gradients method for each example.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels

    :param model_object: Trained neural net (instance of `keras.models.Model`
        or `keras.models.Sequential`).
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param target_class: Array index for class we care about.
    :param num_interp_steps: Number of interpolation steps between baseline
        image (all zeros) and actual image, to be used for each example.
    :return: integ_gradient_matrix: E-by-M-by-N-by-C numpy array of integrated
        gradients.
    """

    assert not numpy.any(numpy.isnan(predictor_matrix))
    assert len(predictor_matrix.shape) == 4

    num_interp_steps = int(numpy.round(num_interp_steps))
    assert num_interp_steps >= 10

    num_examples = predictor_matrix.shape[0]
    integ_gradient_matrix = numpy.full(predictor_matrix.shape, numpy.nan)

    for i in range(num_examples):
        if numpy.mod(i, 5) == 0:
            print((
                'Have run integrated-gradients method for {0:d} of {1:d} '
                'examples...'
            ).format(
                i, num_examples
            ))

        this_predictor_matrix_interp = _interp_predictors_one_example(
            predictor_matrix=predictor_matrix[i, ...],
            num_steps=num_interp_steps
        )

        this_gradient_matrix = _compute_gradients(
            model_object=model_object,
            predictor_matrix_interp=this_predictor_matrix_interp,
            target_class=target_class
        )

        integ_gradient_matrix[i, ...] = _accumulate_gradients(
            predictor_matrix_actual=predictor_matrix[i, ...],
            gradient_matrix=this_gradient_matrix
        )

    print('Ran integrated-gradients method for all {0:d} examples!'.format(
        num_examples
    ))

    return integ_gradient_matrix
