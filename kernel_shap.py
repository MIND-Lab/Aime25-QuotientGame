import numpy as np
import itertools
import warnings
import random
import xai_mind.shap.utils as utils
import copy

def compute(n_samples, background_dataset, instance_to_explain, predict_function, classes):
    n_features = background_dataset.shape[1]
    max_features = 10
    if n_features <= max_features:
        max_samples = 2 ** n_features - 2
        if n_samples > max_samples:
            n_samples = max_samples
    mask = utils.sampling(n_features, n_samples)
    masked_dataset = utils.convert_to_original_feature_space(background_dataset,
                                                              mask,
                                                              instance_to_explain)

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
    nonzero_inds = np.arange(n_features)

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

        phi = np.zeros(n_features)

        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (fx_single - random_prediction_single) - sum(w) #fx - random_prediction = fx(z') - sum(w) =fx(x)
        all_phis.append(phi)

    return all_phis
