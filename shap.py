import keras.saving
import tensorflow as tf
import numpy as np
import os
import gc
import psutil

import kernel_shap as ks


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

def leave_one_patient_out(base_path, ds, output_path, bc_sample_number=100,
                          n_samples=256, classes=[0,1,2]):

    tf.keras.backend.clear_session()

    real_v = []
    heatmaps = []
    int_predictions = []

    folds = ds.leave_one_patient_cv()
    for j, (train_idx, test_idx) in enumerate(folds):
        names_test_cv = ds.user[test_idx]
        print(names_test_cv[0])
        model = keras.saving.load_model(base_path+str(np.unique(names_test_cv))+'.keras')
        X_train_cv = ds.spectra[train_idx]
        y_train_cv = ds.labels[train_idx]
        X_test_cv = ds.spectra[test_idx]
        y_test_cv = ds.labels[test_idx]
        bc_ds = sample_for_background_dataset(X_train_cv, y_train_cv, bc_sample_number)
        for i in range(len(X_test_cv)):
            test_el = X_test_cv[i]
            pred_val = model.test_model(np.array([test_el]))
            int_predictions.append(pred_val)
            sl = ks.kernel_shap(n_samples, bc_ds, test_el, model.model.predict, classes)
            heatmaps.append(sl)
            real_v.append(y_test_cv[i])
            gc.collect()
        print(psutil.virtual_memory().percent)
        if psutil.virtual_memory().percent > 95:
            exit()
        print(psutil.cpu_percent())
        gc.collect()

    os.makedirs(output_path, exist_ok=True)
    heatmaps_new_shap = np.array(heatmaps)
    np.save(output_path + 'shap.npy', heatmaps_new_shap)
    real_v = np.array(real_v)
    np.save(output_path + 'real_labels.npy', real_v)
    int_predictions = np.array(int_predictions)
    np.save(output_path + 'predictions.npy', int_predictions)

def single_test(model_path, X_Train, X_Test, y_test, y_train, output_path, bc_sample_number=100,
                          n_samples=256, classes=[0,1,2]):
    tf.keras.backend.clear_session()

    heatmaps = []
    real_v = []

    model = tf.keras.saving.load_model(model_path)

    predictions = model.test_model(X_Test)
    bc_ds = sample_for_background_dataset(X_Train, y_train, bc_sample_number)
    for i in range(len(X_Test)):
        test_el = X_Test[i]
        sl = ks.kernel_shap(n_samples, bc_ds, test_el, model.predict, classes)
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
