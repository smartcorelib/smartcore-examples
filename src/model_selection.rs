use smartcore::dataset::breast_cancer;
use smartcore::dataset::diabetes;
use smartcore::dataset::iris;
use std::fs::File;
use std::io::prelude::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Imports for KNN classifier
use smartcore::math::distance::*;
use smartcore::neighbors::knn_classifier::*;
// Linear regression
use smartcore::linear::linear_regression::LinearRegression;
// Logistic regression
use smartcore::linear::logistic_regression::LogisticRegression;
// Model performance
use smartcore::metrics::accuracy;
// K-fold CV
use smartcore::model_selection::{cross_val_predict, cross_validate, KFold};

use crate::utils;

pub fn save_restore_knn() {
    // Load Iris dataset
    let iris_data = iris::load_dataset();
    // Turn Iris dataset into NxM matrix
    let x = DenseMatrix::from_array(
        iris_data.num_samples,
        iris_data.num_features,
        &iris_data.data,
    );
    // These are our target class labels
    let y = iris_data.target;

    // Fit KNN classifier to Iris dataset
    let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();
    // File name for the model
    let file_name = "iris_knn.model";
    // Save the model
    {
        let knn_bytes = bincode::serialize(&knn).expect("Can not serialize the model");
        File::create(file_name)
            .and_then(|mut f| f.write_all(&knn_bytes))
            .expect("Can not persist model");
    }
    // Load the model
    let knn: KNNClassifier<f32, euclidian::Euclidian> = {
        let mut buf: Vec<u8> = Vec::new();
        File::open(&file_name)
            .and_then(|mut f| f.read_to_end(&mut buf))
            .expect("Can not load model");
        bincode::deserialize(&buf).expect("Can not deserialize the model")
    };
    //Predict class labels
    let y_hat = knn.predict(&x).unwrap(); // Predict class labels
                                          // Calculate training error
    println!("accuracy: {}", accuracy(&y, &y_hat)); // Prints 0.96
}

// This example is expired by
// https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html
pub fn lr_cross_validate() {
    // Load dataset
    let breast_cancer_data = breast_cancer::load_dataset();
    let x = DenseMatrix::from_array(
        breast_cancer_data.num_samples,
        breast_cancer_data.num_features,
        &breast_cancer_data.data,
    );
    // These are our target values
    let y = breast_cancer_data.target;
    // cross-validated estimator
    let results = cross_validate(
        LogisticRegression::fit,
        &x,
        &y,
        Default::default(),
        KFold::default().with_n_splits(3),
        accuracy,
    )
    .unwrap();
    println!(
        "Test score: {}, training score: {}",
        results.mean_test_score(),
        results.mean_train_score()
    );
}

// This example is expired by
// https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html
pub fn plot_cross_val_predict() {
    // Load Diabetes dataset
    let diabetes_data = diabetes::load_dataset();
    let x = DenseMatrix::from_array(
        diabetes_data.num_samples,
        diabetes_data.num_features,
        &diabetes_data.data,
    );
    // These are our target values
    let y = diabetes_data.target;
    // Generate cross-validated estimates for each input data point
    let y_hat = cross_val_predict(
        LinearRegression::fit,
        &x,
        &y,
        Default::default(),
        KFold::default().with_n_splits(10),
    )
    .unwrap();
    // Assemble XY dataset for the scatter plot
    let xy = DenseMatrix::from_2d_vec(
        &y_hat
            .into_iter()
            .zip(y.into_iter())
            .map(|(x1, x2)| vec![x1, x2])
            .collect(),
    );
    // Plot XY
    utils::scatterplot(&xy, None, "cross_val_predict").unwrap();
}
