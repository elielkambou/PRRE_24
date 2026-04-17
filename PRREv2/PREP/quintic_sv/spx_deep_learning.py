from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .curves import build_spline_forward_curve_from_anchor_values, sample_forward_curve_on_anchor_grid
from .spx_monte_carlo import price_spx_smile_with_monte_carlo
from .types import (
    ConstantHModelParameters,
    ExperimentInput,
    MonteCarloSettings,
    ParametricForwardVarianceCurve,
    QuinticPolynomialCoefficients,
    SmileGridSettings,
    SPXSmileResult,
    SplineForwardVarianceCurve,
    TimeDependentHModelParameters,
)
from .utils import default_spx_log_moneyness_range
from .black import implied_volatility_vector


SURROGATE_FORWARD_ANCHOR_TIMES = np.array([0.0, 0.03, 0.08, 0.16, 0.25, 0.40, 0.70, 1.20], dtype=float)


@dataclass
class SurrogateDataset:
    """Features and targets used to train the SPX surrogate."""

    features: np.ndarray
    targets: np.ndarray
    feature_names: list[str]


@dataclass
class TrainingHistory:
    """Loss curves tracked during neural training."""

    train_loss: list[float]
    validation_loss: list[float]


@dataclass
class SPXSurrogateModel:
    """Portable MLP model stored entirely as Numpy arrays."""

    forward_anchor_times: np.ndarray
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    weights: list[np.ndarray]
    biases: list[np.ndarray]


def build_surrogate_feature_names(anchor_times: np.ndarray = SURROGATE_FORWARD_ANCHOR_TIMES) -> list[str]:
    """Return the ordered feature names used by the MLP."""
    base_names = [
        "is_time_dependent_h",
        "rho",
        "epsilon",
        "alpha0",
        "alpha1",
        "alpha3",
        "alpha5",
        "h0",
        "h_inf",
        "h_kappa",
        "maturity",
        "log_moneyness",
    ]
    curve_names = [f"forward_anchor_{index}" for index in range(len(anchor_times))]
    return base_names + curve_names


def build_spx_surrogate_feature_vector(
    experiment: ExperimentInput,
    maturity: float,
    log_moneyness: float,
    anchor_times: np.ndarray = SURROGATE_FORWARD_ANCHOR_TIMES,
) -> np.ndarray:
    """Turn one option query into a fixed-size vector for the MLP."""
    if isinstance(experiment.model_parameters, ConstantHModelParameters):
        is_time_dependent_h = 0.0
        h0 = experiment.model_parameters.h_value
        h_inf = experiment.model_parameters.h_value
        h_kappa = 0.0
    else:
        is_time_dependent_h = 1.0
        h0 = experiment.model_parameters.h0
        h_inf = experiment.model_parameters.h_inf
        h_kappa = experiment.model_parameters.h_kappa

    coefficients = experiment.model_parameters.polynomial
    forward_curve_features = sample_forward_curve_on_anchor_grid(experiment.forward_curve, anchor_times)
    return np.concatenate(
        [
            np.array(
                [
                    is_time_dependent_h,
                    experiment.model_parameters.rho,
                    experiment.model_parameters.epsilon,
                    coefficients.alpha0,
                    coefficients.alpha1,
                    coefficients.alpha3,
                    coefficients.alpha5,
                    h0,
                    h_inf,
                    h_kappa,
                    maturity,
                    log_moneyness,
                ],
                dtype=float,
            ),
            forward_curve_features.astype(float),
        ]
    )


def sample_random_polynomial_coefficients(rng: np.random.Generator) -> QuinticPolynomialCoefficients:
    """Sample positive polynomial coefficients in a paper-like range."""
    return QuinticPolynomialCoefficients(
        alpha0=float(rng.uniform(0.0, 1.0)),
        alpha1=float(rng.uniform(0.05, 1.2)),
        alpha3=float(rng.uniform(0.0, 0.40)),
        alpha5=float(rng.uniform(0.0, 0.50)),
    )


def sample_random_spline_forward_curve(rng: np.random.Generator) -> SplineForwardVarianceCurve:
    """Sample a smooth positive spline curve for training data."""
    anchor_times = SURROGATE_FORWARD_ANCHOR_TIMES
    normalized_time = anchor_times / max(anchor_times[-1], 1e-12)
    base_level = rng.uniform(0.006, 0.030)
    slope = rng.uniform(-0.010, 0.018)
    hump_height = rng.uniform(-0.008, 0.015)
    hump_center = rng.uniform(0.08, 0.70)
    hump_width = rng.uniform(0.05, 0.35)
    noise = rng.normal(loc=0.0, scale=0.0015, size=anchor_times.shape)
    values = (
        base_level
        + slope * normalized_time
        + hump_height * np.exp(-((anchor_times - hump_center) ** 2) / (2.0 * hump_width**2))
        + noise
    )
    return build_spline_forward_curve_from_anchor_values(anchor_times, np.clip(values, 0.0025, 0.0800))


def sample_random_parametric_forward_curve(rng: np.random.Generator) -> ParametricForwardVarianceCurve:
    """Sample a paper-like parametric curve xi_0(t) = a e^{-b t} + c (1 - e^{-b t})."""
    return ParametricForwardVarianceCurve(
        a=float(rng.uniform(0.004, 0.025)),
        b=float(rng.uniform(0.4, 4.0)),
        c=float(rng.uniform(0.010, 0.055)),
    )


def build_random_teacher_experiment(
    rng: np.random.Generator,
    teacher_steps: int,
    teacher_base_paths: int,
    allow_time_dependent_h: bool = True,
) -> ExperimentInput:
    """Sample a random input configuration and package it as ExperimentInput."""
    polynomial = sample_random_polynomial_coefficients(rng)
    if allow_time_dependent_h and rng.random() < 0.30:
        model_parameters = TimeDependentHModelParameters(
            rho=float(rng.uniform(-0.95, -0.05)),
            h0=float(rng.uniform(-0.20, 0.30)),
            h_inf=float(rng.uniform(-0.20, 0.30)),
            h_kappa=float(rng.uniform(0.5, 8.0)),
            epsilon=float(rng.uniform(0.02, 0.20)),
            polynomial=polynomial,
        )
    else:
        model_parameters = ConstantHModelParameters(
            rho=float(rng.uniform(-0.95, -0.05)),
            h_value=float(rng.uniform(-0.20, 0.30)),
            epsilon=float(rng.uniform(0.02, 0.20)),
            polynomial=polynomial,
        )

    if rng.random() < 0.50:
        forward_curve = sample_random_spline_forward_curve(rng)
    else:
        forward_curve = sample_random_parametric_forward_curve(rng)

    return ExperimentInput(
        name="random_teacher",
        spot=100.0,
        model_parameters=model_parameters,
        forward_curve=forward_curve,
        smile_grid=SmileGridSettings(
            spx_maturities=np.array([0.10], dtype=float),
            vix_maturities=np.array([0.10], dtype=float),
        ),
        monte_carlo=MonteCarloSettings(
            spx_n_steps=teacher_steps,
            spx_n_base_paths=teacher_base_paths,
            vix_n_steps=50,
            quadrature_degree=100,
            random_seed=int(rng.integers(1, 1_000_000)),
        ),
    )


def generate_spx_surrogate_dataset(
    number_of_experiments: int,
    strikes_per_experiment: int,
    teacher_steps: int = 64,
    teacher_base_paths: int = 600,
    seed: int = 0,
    allow_time_dependent_h: bool = True,
) -> SurrogateDataset:
    """Generate a regression dataset using Monte Carlo as teacher."""
    rng = np.random.default_rng(seed)
    feature_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    feature_names = build_surrogate_feature_names()

    for _ in range(number_of_experiments):
        experiment = build_random_teacher_experiment(
            rng,
            teacher_steps=teacher_steps,
            teacher_base_paths=teacher_base_paths,
            allow_time_dependent_h=allow_time_dependent_h,
        )
        maturity = float(rng.uniform(7.0 / 365.0, 1.20))
        log_moneyness_min, log_moneyness_max = default_spx_log_moneyness_range(maturity)
        log_moneyness_grid = np.linspace(log_moneyness_min, log_moneyness_max, strikes_per_experiment)
        strikes = experiment.spot * np.exp(log_moneyness_grid)
        smile = price_spx_smile_with_monte_carlo(experiment, maturity, strikes)

        for log_moneyness, option_price in zip(log_moneyness_grid, smile.option_prices, strict=True):
            feature_rows.append(build_spx_surrogate_feature_vector(experiment, maturity, float(log_moneyness)))
            target_rows.append(float(option_price / experiment.spot))

    return SurrogateDataset(
        features=np.vstack(feature_rows),
        targets=np.asarray(target_rows, dtype=float),
        feature_names=feature_names,
    )


def fit_feature_standardization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std with protection against zero variance."""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return mean, np.where(std < 1e-8, 1.0, std)


def standardize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply standard score normalization."""
    return (features - mean) / std


def initialize_dense_network(layer_sizes: list[int], rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """He-initialize a ReLU MLP."""
    weights = []
    biases = []
    for input_dim, output_dim in zip(layer_sizes[:-1], layer_sizes[1:], strict=True):
        scale = np.sqrt(2.0 / input_dim)
        weights.append(rng.normal(loc=0.0, scale=scale, size=(input_dim, output_dim)))
        biases.append(np.zeros(output_dim, dtype=float))
    return weights, biases


def relu_activation(values: np.ndarray) -> np.ndarray:
    """ReLU non-linearity."""
    return np.maximum(values, 0.0)


def forward_dense_network(
    inputs: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run a forward pass and keep all intermediates for backpropagation."""
    activations = [inputs]
    pre_activations = []
    current = inputs

    for weight, bias in zip(weights[:-1], biases[:-1], strict=True):
        z_values = current @ weight + bias
        pre_activations.append(z_values)
        current = relu_activation(z_values)
        activations.append(current)

    final_z = current @ weights[-1] + biases[-1]
    pre_activations.append(final_z)
    activations.append(final_z)
    return activations, pre_activations


def backward_dense_network(
    weights: list[np.ndarray],
    activations: list[np.ndarray],
    pre_activations: list[np.ndarray],
    targets: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Backpropagate the MSE loss through the dense network."""
    batch_size = targets.shape[0]
    delta = 2.0 * (activations[-1] - targets.reshape(-1, 1)) / batch_size
    grad_weights = [np.zeros_like(weight) for weight in weights]
    grad_biases = [np.zeros(weight.shape[1], dtype=float) for weight in weights]

    for layer_index in reversed(range(len(weights))):
        grad_weights[layer_index] = activations[layer_index].T @ delta
        grad_biases[layer_index] = np.sum(delta, axis=0)
        if layer_index > 0:
            delta = (delta @ weights[layer_index].T) * (pre_activations[layer_index - 1] > 0.0)

    return grad_weights, grad_biases


def initialize_adam_state(parameters: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Create zero first and second moments for Adam."""
    return [np.zeros_like(parameter) for parameter in parameters], [np.zeros_like(parameter) for parameter in parameters]


def apply_adam_update(
    parameters: list[np.ndarray],
    gradients: list[np.ndarray],
    first_moment: list[np.ndarray],
    second_moment: list[np.ndarray],
    learning_rate: float,
    step_index: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> None:
    """In-place Adam update."""
    for parameter, gradient, moment_1, moment_2 in zip(parameters, gradients, first_moment, second_moment, strict=True):
        moment_1 *= beta1
        moment_1 += (1.0 - beta1) * gradient
        moment_2 *= beta2
        moment_2 += (1.0 - beta2) * (gradient**2)

        corrected_1 = moment_1 / (1.0 - beta1**step_index)
        corrected_2 = moment_2 / (1.0 - beta2**step_index)
        parameter -= learning_rate * corrected_1 / (np.sqrt(corrected_2) + epsilon)


def predict_standardized_targets(
    features: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> np.ndarray:
    """Inference helper returning the raw standardized network output."""
    current = features
    for weight, bias in zip(weights[:-1], biases[:-1], strict=True):
        current = relu_activation(current @ weight + bias)
    return (current @ weights[-1] + biases[-1]).reshape(-1)


def train_spx_surrogate_model(
    dataset: SurrogateDataset,
    hidden_layer_sizes: tuple[int, ...] = (64, 64, 64),
    epochs: int = 250,
    batch_size: int = 128,
    learning_rate: float = 3e-3,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[SPXSurrogateModel, TrainingHistory]:
    """Train a dense ReLU regressor on Monte Carlo SPX labels."""
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(dataset.features.shape[0])
    split_index = max(1, int(round((1.0 - validation_fraction) * dataset.features.shape[0])))
    train_indices = shuffled_indices[:split_index]
    validation_indices = shuffled_indices[split_index:]

    train_features = dataset.features[train_indices]
    validation_features = dataset.features[validation_indices] if len(validation_indices) > 0 else dataset.features[train_indices]
    train_targets = dataset.targets[train_indices]
    validation_targets = dataset.targets[validation_indices] if len(validation_indices) > 0 else dataset.targets[train_indices]

    feature_mean, feature_std = fit_feature_standardization(train_features)
    target_mean = float(np.mean(train_targets))
    target_std = float(max(np.std(train_targets), 1e-8))

    train_features_scaled = standardize_features(train_features, feature_mean, feature_std)
    validation_features_scaled = standardize_features(validation_features, feature_mean, feature_std)
    train_targets_scaled = (train_targets - target_mean) / target_std
    validation_targets_scaled = (validation_targets - target_mean) / target_std

    layer_sizes = [train_features_scaled.shape[1], *hidden_layer_sizes, 1]
    weights, biases = initialize_dense_network(layer_sizes, rng)
    weight_m1, weight_m2 = initialize_adam_state(weights)
    bias_m1, bias_m2 = initialize_adam_state(biases)

    train_loss_history: list[float] = []
    validation_loss_history: list[float] = []
    adam_step = 0

    for _ in range(epochs):
        batch_permutation = rng.permutation(train_features_scaled.shape[0])
        for batch_start in range(0, train_features_scaled.shape[0], batch_size):
            batch_indices = batch_permutation[batch_start : batch_start + batch_size]
            batch_features = train_features_scaled[batch_indices]
            batch_targets = train_targets_scaled[batch_indices]
            activations, pre_activations = forward_dense_network(batch_features, weights, biases)
            grad_weights, grad_biases = backward_dense_network(weights, activations, pre_activations, batch_targets)
            adam_step += 1
            apply_adam_update(weights, grad_weights, weight_m1, weight_m2, learning_rate, adam_step)
            apply_adam_update(biases, grad_biases, bias_m1, bias_m2, learning_rate, adam_step)

        train_predictions = predict_standardized_targets(train_features_scaled, weights, biases)
        validation_predictions = predict_standardized_targets(validation_features_scaled, weights, biases)
        train_loss_history.append(float(np.mean((train_predictions - train_targets_scaled) ** 2)))
        validation_loss_history.append(float(np.mean((validation_predictions - validation_targets_scaled) ** 2)))

    model = SPXSurrogateModel(
        forward_anchor_times=SURROGATE_FORWARD_ANCHOR_TIMES.copy(),
        feature_names=list(dataset.feature_names),
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        weights=weights,
        biases=biases,
    )
    history = TrainingHistory(train_loss=train_loss_history, validation_loss=validation_loss_history)
    return model, history


def predict_normalized_spx_prices(model: SPXSurrogateModel, features: np.ndarray) -> np.ndarray:
    """Predict call prices normalized by spot."""
    scaled_features = standardize_features(np.asarray(features, dtype=float), model.feature_mean, model.feature_std)
    standardized_predictions = predict_standardized_targets(scaled_features, model.weights, model.biases)
    normalized_prices = standardized_predictions * model.target_std + model.target_mean
    return np.clip(normalized_prices, 0.0, 1.0)


def price_spx_smile_with_surrogate(
    experiment: ExperimentInput,
    maturity: float,
    strikes: np.ndarray,
    model: SPXSurrogateModel,
) -> SPXSmileResult:
    """Replace the SPX Monte Carlo by a trained neural surrogate."""
    strikes_array = np.asarray(strikes, dtype=float)
    log_moneyness = np.log(strikes_array / experiment.spot)
    feature_matrix = np.vstack(
        [
            build_spx_surrogate_feature_vector(experiment, maturity, float(log_moneyness_value), model.forward_anchor_times)
            for log_moneyness_value in log_moneyness
        ]
    )
    normalized_prices = predict_normalized_spx_prices(model, feature_matrix)
    option_prices = normalized_prices * experiment.spot
    return SPXSmileResult(
        maturity=maturity,
        strikes=strikes_array,
        log_moneyness=log_moneyness,
        option_prices=option_prices,
        implied_volatility=implied_volatility_vector(option_prices, experiment.spot, strikes_array, maturity),
        engine_name="deep_learning",
        confidence_interval_half_width=None,
    )


def save_spx_surrogate_model(model: SPXSurrogateModel, output_path: str | Path) -> Path:
    """Serialize a surrogate model to an .npz file."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "forward_anchor_times": np.asarray(model.forward_anchor_times, dtype=float),
        "feature_mean": np.asarray(model.feature_mean, dtype=float),
        "feature_std": np.asarray(model.feature_std, dtype=float),
        "target_mean": np.array([model.target_mean], dtype=float),
        "target_std": np.array([model.target_std], dtype=float),
        "feature_names": np.asarray(model.feature_names, dtype=str),
        "layer_count": np.array([len(model.weights)], dtype=int),
    }
    for layer_index, (weight, bias) in enumerate(zip(model.weights, model.biases, strict=True)):
        payload[f"weight_{layer_index}"] = weight
        payload[f"bias_{layer_index}"] = bias
    np.savez_compressed(destination, **payload)
    return destination


def load_spx_surrogate_model(model_path: str | Path) -> SPXSurrogateModel:
    """Load a surrogate model saved with save_spx_surrogate_model."""
    saved = np.load(Path(model_path))
    layer_count = int(saved["layer_count"][0])
    weights = [saved[f"weight_{index}"] for index in range(layer_count)]
    biases = [saved[f"bias_{index}"] for index in range(layer_count)]
    return SPXSurrogateModel(
        forward_anchor_times=saved["forward_anchor_times"],
        feature_names=list(saved["feature_names"].astype(str)),
        feature_mean=saved["feature_mean"],
        feature_std=saved["feature_std"],
        target_mean=float(saved["target_mean"][0]),
        target_std=float(saved["target_std"][0]),
        weights=weights,
        biases=biases,
    )
