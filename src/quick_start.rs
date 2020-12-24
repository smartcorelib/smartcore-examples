use smartcore::dataset::iris::load_dataset;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// ndarray structs
use ndarray::Array;
// nalgebra structs
use nalgebra::{DMatrix, RowDVector};
// Imports for KNN classifier
use smartcore::neighbors::knn_classifier::*;
// Imports for Logistic Regression
use smartcore::linear::logistic_regression::LogisticRegression;
// Imports Gaussian Naive Bayes classifier
use smartcore::naive_bayes::gaussian::GaussianNB;
// Model performance
use smartcore::metrics::accuracy;

pub fn iris_knn_example() {
    // Load Iris dataset
    let iris_data = load_dataset();
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

    let y_hat = knn.predict(&x).unwrap(); // Predict class labels

    // Calculate training error
    println!("accuracy: {}", accuracy(&y, &y_hat)); // Prints 0.96
}

pub fn iris_lr_example() {
    // Load Iris dataset
    let iris_data = load_dataset();
    // Turn Iris dataset into NxM matrix
    let x = DenseMatrix::from_array(
        iris_data.num_samples,
        iris_data.num_features,
        &iris_data.data,
    );
    // These are our target class labels
    let y = iris_data.target;

    // Fit Logistic Regression to Iris dataset
    let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
    let y_hat = lr.predict(&x).unwrap(); // Predict class labels

    // Calculate training error
    println!("accuracy: {}", accuracy(&y, &y_hat)); // Prints 0.98
}

pub fn iris_lr_ndarray_example() {
    // Load Iris dataset
    let iris_data = load_dataset();
    // Turn Iris dataset into NxM matrix
    let x = Array::from_shape_vec(
        (iris_data.num_samples, iris_data.num_features),
        iris_data.data,
    )
    .unwrap();

    // These are our target class labels
    let y = Array::from_shape_vec(iris_data.num_samples, iris_data.target).unwrap();

    // Fit Logistic Regression to Iris dataset
    let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
    let y_hat = lr.predict(&x).unwrap(); // Predict class labels

    // Calculate training error
    println!("accuracy: {}", accuracy(&y, &y_hat)); // Prints 0.98
}

pub fn iris_lr_nalgebra_example() {
    // Load Iris dataset
    let iris_data = load_dataset();
    // Turn Iris dataset into NxM matrix
    let x = DMatrix::from_row_slice(
        iris_data.num_samples,
        iris_data.num_features,
        &iris_data.data,
    );

    // These are our target class labels
    let y = RowDVector::from_iterator(iris_data.num_samples, iris_data.target.into_iter());

    // Fit Logistic Regression to Iris dataset
    let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
    let y_hat = lr.predict(&x).unwrap(); // Predict class labels

    // Calculate training error
    println!("accuracy: {}", accuracy(&y, &y_hat)); // Prints 0.98
}

pub fn iris_gaussiannb_example() {
    // Load Iris dataset
    let iris_data = load_dataset();

    // Turn Iris dataset into NxM matrix
    let x = DenseMatrix::from_array(
        iris_data.num_samples,
        iris_data.num_features,
        &iris_data.data,
    );

    // These are our target class labels
    let y = iris_data.target;

    // Fit Logistic Regression to Iris dataset
    let gnb = GaussianNB::fit(&x, &y, Default::default()).unwrap();
    let y_hat = gnb.predict(&x).unwrap(); // Predict class labels

    // Calculate training error
    println!("accuracy: {}", accuracy(&y, &y_hat)); // Prints 0.96
}
