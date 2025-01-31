import tensorflow as tf
import numpy as np
import GradCam as gcam_base
import os
import cnn_models


def leave_one_patient_out(base_path, ds, output_path, input_shape=991, output_shape=3, v_size=850):

    tf.keras.backend.clear_session()
    base_model = cnn_models.load_model_benchmark_functional(
        n_dims=input_shape, number_classes=output_shape)

    folds = ds.leave_one_patient_cv()

    heatmaps_global = []
    predictions_global = []
    real_value_global = []

    for j, (train_idx, test_idx) in enumerate(folds):
        names_test_cv = ds.user[test_idx]
        print(names_test_cv[0])
        base_model.load_weights(base_path+str(np.unique(names_test_cv))+'.weights.h5')
        X_test_cv = ds.spectra[test_idx]
        y_test_cv = ds.labels[test_idx]
        predictions = base_model.test_model(X_test_cv)
        gCam = gcam_base.GradCam(X_test_cv, base_model)
        heatMaps = gCam.make_gradcam_heatmap('conv1d_2')
        heatmaps_new = []
        for heatmap in heatMaps:
            heat = gCam.cubic_spline_interpolation(heatmap, v_size, input_shape)
            heatmaps_new.append(np.squeeze(heat, 1))
        heatmaps_new = np.array(heatmaps_new)
        heatmaps_global.extend(heatmaps_new)
        predictions_global.extend(predictions)
        real_value_global.extend(y_test_cv)

    os.makedirs(output_path, exist_ok=True)
    heatmaps_global = np.array(heatmaps_global)
    np.save(output_path+"heatmaps.npy", heatmaps_global)
    predictions_global = np.array(predictions_global)
    np.save(output_path + "predictions.npy", predictions_global)
    real_value_global = np.array(real_value_global)
    np.save(output_path + "real_values.npy", real_value_global)


def single_test(model_path, X_Test, y_test, output_path, input_shape=991, output_shape=3,v_size=850):
    tf.keras.backend.clear_session()
    base_model = cnn_models.load_model_benchmark(n_dims=input_shape, number_classes=output_shape)

    base_model.load_weights(model_path)
    predictions = base_model.test_model(X_Test)
    gCam = gcam_base.GradCam(X_Test, base_model)
    heatMaps = gCam.make_gradcam_heatmap('conv1d_2')
    heatmaps_new = []
    for heatmap in heatMaps:
        heat = gCam.cubic_spline_interpolation(heatmap, v_size, input_shape)
        heatmaps_new.append(heat)
    heatmaps_new = np.array(heatmaps_new)

    os.makedirs(output_path, exist_ok=True)
    heatmaps_global = np.array(heatmaps_new)
    np.save(output_path + "heatmaps.npy", heatmaps_global)
    predictions_global = np.array(predictions)
    np.save(output_path + "predictions.npy", predictions_global)
    real_value_global = np.array(y_test)
    np.save(output_path + "real_values.npy", real_value_global)
