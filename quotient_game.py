import numpy as np
import copy
from scipy.special import binom
from scipy.special import comb
import itertools
import random
import math
import xai_mind.shap.utils as utils

def mask_to_ps(mask, ps):
    new_mask = []
    for m in mask:
        new_m = np.zeros(len(ps))
        value_with_one = np.where(m == 1)[0]
        for v in value_with_one:
            indexes = np.where(ps == v)[0]
            new_m[indexes[0]:indexes[-1]+1] = 1
        new_mask.append(new_m)
    new_mask = np.array(new_mask)
    return new_mask

def transform_phis_to_original_dimension(phi, ps):
    new_phi = []
    for i in range(len(phi)):
        for p in ps:
            if p == i:
                new_phi.append(phi[i])
    return np.array(new_phi)

def generate_ps(n_features, n_players):
    ps = []
    window_dimension = math.floor(n_features / n_players)
    diff_points = n_features - (window_dimension * n_players)
    if diff_points == 0:
        last_window_size = window_dimension
    else:
        if diff_points > window_dimension / 2:
            n_players += 1
            last_window_size = diff_points
        else:
            last_window_size = window_dimension + diff_points
    for i in range(n_players-1):
        for _ in range(window_dimension):
            ps.append(i)
    for _ in range(last_window_size):
        ps.append(n_players-1)
    return np.array(ps)

def compute(n_samples, background_dataset, instance_to_explain, predict_function, classes, n_players, ps=[],
                substitution_strategy = "default", sigma=0.5, mean=0):
    n_features = background_dataset.shape[1]
    max_features = 15
    if n_features <= max_features:
        max_samples = 2 ** n_players - 2
        if n_samples > max_samples:
            n_samples = max_samples
    if len(ps) == 0:
        ps = generate_ps(n_features, n_players)
    if len(ps) != n_features:
        raise Exception("Dimension of the Partition Structure should match the number of features")
    mask = utils.sampling(n_players, n_samples)
    new_mask = mask_to_ps(mask, ps)
    if substitution_strategy == "default":
        masked_dataset = utils.convert_to_original_feature_space(background_dataset,
                                                                      new_mask,
                                                                      instance_to_explain)
    elif substitution_strategy == "gaussian":
        masked_dataset = utils.convert_to_original_features_space_gaussian(new_mask,
                                                                     instance_to_explain,
                                                                     sigma,
                                                                     mean)
    else:
        raise Exception("Substitution strategy not recognized!")

    weights = []
    for m in mask:
        w = utils.calculate_weights(m)
        weights.append(w)
    weights = np.array(weights)
    weights /= np.sum(weights)

    random_prediction = np.mean(predict_function(background_dataset), axis=0)

    predictions = []
    for k in range(len(mask)):
        ms = masked_dataset[k]
        pred = predict_function(ms)
        it_pred = np.sum(pred, axis=0) * 1/(len(background_dataset))
        predictions.append(it_pred)
    predictions = np.array(predictions)

    fx = predict_function(np.array([instance_to_explain]))[0]

    eyAdj = predictions - random_prediction
    nonzero_inds = np.arange(n_players)

    all_phis = []
    for c in classes:
        fx_single = fx[c]
        random_prediction_single = random_prediction[c]

        eyAdj2 = eyAdj[:,c] - mask[:, nonzero_inds[-1]] * (fx_single - random_prediction_single)

        etmp = np.transpose(np.transpose(mask[:, nonzero_inds[:-1]]) - mask[:, nonzero_inds[-1]])

        y = eyAdj2

        X = etmp
        WX = weights[:, None] * X

        try:
            w = np.linalg.solve(X.T @ WX, WX.T @ y)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Linear regression equation is singular, a least squares solutions is used instead.\n"
                "To avoid this situation and get a regular matrix do one of the following:\n"
                "1) turn up the number of samples,\n"
                "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
                "3) group features together to reduce the number of inputs that need to be explained."
            )
            sqrt_W = np.sqrt(weights)
            w = np.linalg.lstsq(sqrt_W[:, None] * X, sqrt_W * y, rcond=None)[0]

        phi = np.zeros(n_players)

        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (fx_single - random_prediction_single) - sum(w) #fx - random_prediction = fx(z') - sum(w) =fx(x)
        transformed_phi = transform_phis_to_original_dimension(phi, ps)
        all_phis.append(transformed_phi)

    return all_phis
