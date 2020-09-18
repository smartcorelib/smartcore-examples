use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// KNN
use smartcore::math::distance::Distances;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::neighbors::knn_regressor::KNNRegressor;
// Logistic/Linear Regression
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::linear::logistic_regression::LogisticRegression;
// Tree
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressor;
// Random Forest
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
// Model performance
use smartcore::metrics::{mean_squared_error, roc_auc_score};
use smartcore::model_selection::train_test_split;

pub fn breast_cancer() {
    // Load dataset
    let cancer_data = breast_cancer::load_dataset();
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        cancer_data.num_samples,
        cancer_data.num_features,
        &cancer_data.data,
    );
    // These are our target class labels
    let y = cancer_data.target;

    // Split dataset into training/test (80%/20%)
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2);

    // KNN classifier
    let y_hat_knn = KNNClassifier::fit(
        &x_train,
        &y_train,
        Distances::euclidian(),
        Default::default(),
    )
    .and_then(|knn| knn.predict(&x_test))
    .unwrap();

    // Logistic Regression
    let y_hat_lr = LogisticRegression::fit(&x_train, &y_train)
        .and_then(|lr| lr.predict(&x_test))
        .unwrap();

    // Decision Tree
    let y_hat_tree = DecisionTreeClassifier::fit(&x_train, &y_train, Default::default())
        .and_then(|tree| tree.predict(&x_test))
        .unwrap();

    // Random Forest
    let y_hat_rf = RandomForestClassifier::fit(&x_train, &y_train, Default::default())
        .and_then(|rf| rf.predict(&x_test))
        .unwrap();

    // Calculate test error
    println!("AUC KNN: {}", roc_auc_score(&y_test, &y_hat_knn));
    println!(
        "AUC Logistic Regression: {}",
        roc_auc_score(&y_test, &y_hat_lr)
    );
    println!("AUC Decision Tree: {}", roc_auc_score(&y_test, &y_hat_tree));
    println!("AUC Random Forest: {}", roc_auc_score(&y_test, &y_hat_rf));
}

pub fn boston() {
    // Load dataset
    let cancer_data = boston::load_dataset();
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        cancer_data.num_samples,
        cancer_data.num_features,
        &cancer_data.data,
    );
    // These are our target class labels
    let y = cancer_data.target;

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2);

    // Fit KNN regressor
    let y_hat_knn = KNNRegressor::fit(
        &x_train,
        &y_train,
        Distances::euclidian(),
        Default::default(),
    )
    .and_then(|knn| knn.predict(&x_test))
    .unwrap();

    // Fit Linear Regression
    let y_hat_lr = LinearRegression::fit(&x_train, &y_train, Default::default())
        .and_then(|lr| lr.predict(&x_test))
        .unwrap();

    // Fit Decision Tree
    let y_hat_tree = DecisionTreeRegressor::fit(&x_train, &y_train, Default::default())
        .and_then(|tree| tree.predict(&x_test))
        .unwrap();

    // Fit Random Forest
    let y_hat_rf = RandomForestRegressor::fit(&x_train, &y_train, Default::default())
        .and_then(|rf| rf.predict(&x_test))
        .unwrap();

    // Calculate test error
    println!("MSE KNN: {}", mean_squared_error(&y_test, &y_hat_knn));
    println!(
        "MSE Logistic Regression: {}",
        mean_squared_error(&y_test, &y_hat_lr)
    );
    println!(
        "MSE Decision Tree: {}",
        mean_squared_error(&y_test, &y_hat_tree)
    );
    println!(
        "MSE Random Forest: {}",
        mean_squared_error(&y_test, &y_hat_rf)
    );
}
