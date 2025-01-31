import tensorflow as tf
import numpy as np
import cnn_models as cnn_models
import os
import gc
import psutil

import quotient_game as pks


def sample_for_background_dataset(x, y, dimension = 100):
    elements_for_each_class = int(dimension / len(np.unique(y)))
    sampling = []
    for c in np.unique(y):
        labels = np.where(y == c)[0]
        for i in range(elements_for_each_class):
            v = np.random.randint(0, len(labels)-1)
            index = labels[v]
            sampling.append(x[index])
    elements_left = dimension - (elements_for_each_class * len(np.unique(y)))
    for i in range(elements_left):
        index = np.random.randint(0, len(x) - 1)
        sampling.append(x[index])
    return np.array(sampling)

def leave_one_patient_out(weights_path, ds, output_path, input_shape=991, output_shape=3, bc_sample_number=100,
                            classes=[0,1,2], n_samples=254, n_partition=8):

    tf.keras.backend.clear_session()
    base_model = cnn_models.load_model_benchmark(n_dims=input_shape, number_classes=output_shape)

    real_v = []
    heatmaps = []
    int_predictions = []

    folds = ds.leave_one_patient_cv()
    for j, (train_idx, test_idx) in enumerate(folds):
        names_test_cv = ds.user[test_idx]
        print(names_test_cv[0])
        base_model.load_weights(
            weights_path + str(names_test_cv[0])).expect_partial()
        X_train_cv = ds.spectra[train_idx]
        y_train_cv = ds.labels[train_idx]
        X_test_cv = ds.spectra[test_idx]
        y_test_cv = ds.labels[test_idx]
        bc_ds = sample_for_background_dataset(X_train_cv, y_train_cv, bc_sample_number)
        for i in range(len(X_test_cv)):
            test_el = X_test_cv[i]
            pred_val = base_model.test_model(np.array([test_el]))
            int_predictions.append(pred_val)
            sl = pks.compute(n_samples,
                                 bc_ds,
                                 test_el,
                                 base_model.model.predict,
                                 classes,
                                 n_partition)
            heatmaps.append(sl)
            real_v.append(y_test_cv[i])
            gc.collect()
        print(psutil.virtual_memory().percent)
        print(psutil.cpu_percent())
        gc.collect()

    os.makedirs(output_path, exist_ok=True)
    heatmaps_new_shap = np.array(heatmaps)
    np.save(output_path + 'shap.npy', heatmaps_new_shap)
    real_v = np.array(real_v)
    np.save(output_path + 'real_labels.npy', real_v)
    int_predictions = np.array(int_predictions)
    np.save(output_path + 'predictions.npy', int_predictions)

def single_test(weights_path, X_Train, X_Test, y_test, y_train, output_path, input_shape=991, output_shape=3, bc_sample_number=100,
                          classes=[0,1,2], n_samples=254, n_partition=8):
    tf.keras.backend.clear_session()
    base_model = cnn_models.load_model_benchmark(n_dims=input_shape, number_classes=output_shape)

    heatmaps = []
    real_v = []

    base_model.load_weights(
        weights_path).expect_partial()

    predictions = base_model.test_model(X_Test)
    bc_ds = sample_for_background_dataset(X_Train, y_train, bc_sample_number)
    for i in range(len(X_Test)):
        test_el = X_Test[i]
        sl = pks.kernel_shap_partition(n_samples, n_partition, bc_ds, test_el, base_model.model.predict, classes)
        heatmaps.append(sl)
        real_v.append(y_test[i])
        gc.collect()


    os.makedirs(output_path, exist_ok=True)
    heatmaps_global = np.array(heatmaps)
    np.save(output_path + "heatmaps.npy", heatmaps_global)
    predictions_global = np.array(predictions)
    np.save(output_path + "predictions.npy", predictions_global)
    real_value_global = np.array(y_test)
    np.save(output_path + "real_values.npy", real_value_global)
