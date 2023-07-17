use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// KNN
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::neighbors::knn_regressor::KNNRegressor;
// Logistic/Linear Regression
use smartcore::linear::elastic_net::{ElasticNet, ElasticNetParameters};
use smartcore::linear::lasso::{Lasso, LassoParameters};
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};
// Tree
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressor;
// Random Forest
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
// SVM
use smartcore::svm::svc::{SVCParameters, SVC};
use smartcore::svm::svr::{SVRParameters, SVR};
use smartcore::svm::Kernels;
// Model performance
use smartcore::metrics::{mean_squared_error, roc_auc_score};
use smartcore::model_selection::train_test_split;

use crate::utils;

pub fn diabetes() {
    // Load dataset
    let diabetes_data = diabetes::load_dataset();
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        diabetes_data.num_samples,
        diabetes_data.num_features,
        &diabetes_data.data,
    );
    // These are our target values
    let y = diabetes_data.target;

    // Split dataset into training/test (80%/20%)
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, None);

    // SVM
    let y_hat_svm = SVR::fit(
        &x_train,
        &y_train,
        SVRParameters::default()
            .with_kernel(Kernels::rbf(0.5))
            .with_c(2000.0)
            .with_eps(10.0),
    )
    .and_then(|svm| svm.predict(&x_test))
    .unwrap();

    println!("{:?}", y_hat_svm);
    println!("{:?}", y_test);

    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_svm));
}

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
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, None);

    // KNN classifier
    let y_hat_knn = KNNClassifier::fit(&x_train, &y_train, Default::default())
        .and_then(|knn| knn.predict(&x_test))
        .unwrap();

    // Logistic Regression
    let y_hat_lr = LogisticRegression::fit(&x_train, &y_train, Default::default())
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

    // SVM
    let y_hat_svm = SVC::fit(&x_train, &y_train, SVCParameters::default().with_c(2.0))
        .and_then(|svm| svm.predict(&x_test))
        .unwrap();

    // Calculate test error
    println!("AUC KNN: {}", roc_auc_score(&y_test, &y_hat_knn));
    println!(
        "AUC Logistic Regression: {}",
        roc_auc_score(&y_test, &y_hat_lr)
    );
    println!("AUC Decision Tree: {}", roc_auc_score(&y_test, &y_hat_tree));
    println!("AUC Random Forest: {}", roc_auc_score(&y_test, &y_hat_rf));
    println!("AUC SVM: {}", roc_auc_score(&y_test, &y_hat_svm));
}

pub fn boston() {
    // Load dataset
    let boston_data = boston::load_dataset();
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        boston_data.num_samples,
        boston_data.num_features,
        &boston_data.data,
    );
    // These are our target values
    let y = boston_data.target;

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, None);

    // KNN regressor
    let y_hat_knn = KNNRegressor::fit(&x_train, &y_train, Default::default())
        .and_then(|knn| knn.predict(&x_test))
        .unwrap();

    // Linear Regression
    let y_hat_lr = LinearRegression::fit(&x_train, &y_train, Default::default())
        .and_then(|lr| lr.predict(&x_test))
        .unwrap();

    // Ridge Regression
    let y_hat_rr = RidgeRegression::fit(
        &x_train,
        &y_train,
        RidgeRegressionParameters::default().with_alpha(0.5),
    )
    .and_then(|rr| rr.predict(&x_test))
    .unwrap();

    // LASSO
    let y_hat_lasso = Lasso::fit(
        &x_train,
        &y_train,
        LassoParameters::default().with_alpha(0.5),
    )
    .and_then(|lr| lr.predict(&x_test))
    .unwrap();

    // Elastic Net
    let y_hat_en = ElasticNet::fit(
        &x_train,
        &y_train,
        ElasticNetParameters::default()
            .with_alpha(0.5)
            .with_l1_ratio(0.5),
    )
    .and_then(|lr| lr.predict(&x_test))
    .unwrap();

    // Decision Tree
    let y_hat_tree = DecisionTreeRegressor::fit(&x_train, &y_train, Default::default())
        .and_then(|tree| tree.predict(&x_test))
        .unwrap();

    // Random Forest
    let y_hat_rf = RandomForestRegressor::fit(&x_train, &y_train, Default::default())
        .and_then(|rf| rf.predict(&x_test))
        .unwrap();

    // Calculate test error
    println!("MSE KNN: {}", mean_squared_error(&y_test, &y_hat_knn));
    println!(
        "MSE Linear Regression: {}",
        mean_squared_error(&y_test, &y_hat_lr)
    );
    println!(
        "MSE Ridge Regression: {}",
        mean_squared_error(&y_test, &y_hat_rr)
    );
    println!("MSE LASSO: {}", mean_squared_error(&y_test, &y_hat_lasso));
    println!(
        "MSE Elastic Net: {}",
        mean_squared_error(&y_test, &y_hat_en)
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

/// Fits Support Vector Classifier (SVC) to generated dataset and plots the decision boundary for three SVC with different kernels.Default
/// The idea for this example is taken from https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
pub fn svm() {
    let num_samples = 100;
    let num_features = 2;

    // Generate a dataset with 100 sample, 2 features in each sample, split into 2 groups
    let data = generator::make_blobs(num_samples, num_features, 2);
    let y: Vec<f32> = data.target;

    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(data.num_samples, data.num_features, &data.data);

    // We also need a 2x2 mesh grid that we will use to plot decision boundaries.
    let mesh = utils::make_meshgrid(&x);

    // SVC with linear kernel
    let linear_svc = SVC::fit(&x, &y, Default::default()).unwrap();

    utils::scatterplot_with_mesh(
        &mesh,
        &linear_svc.predict(&mesh).unwrap(),
        &x,
        &y,
        "linear_svm",
    )
    .unwrap();

    // SVC with Gaussian kernel
    let rbf_svc = SVC::fit(
        &x,
        &y,
        SVCParameters::default().with_kernel(Kernels::rbf(0.7)),
    )
    .unwrap();

    utils::scatterplot_with_mesh(&mesh, &rbf_svc.predict(&mesh).unwrap(), &x, &y, "rbf_svm")
        .unwrap();

    // SVC with 3rd degree polynomial kernel
    let poly_svc = SVC::fit(
        &x,
        &y,
        SVCParameters::default().with_kernel(Kernels::polynomial_with_degree(3.0, num_features)),
    )
    .unwrap();

    utils::scatterplot_with_mesh(
        &mesh,
        &poly_svc.predict(&mesh).unwrap(),
        &x,
        &y,
        "polynomial_svm",
    )
    .unwrap();
}
