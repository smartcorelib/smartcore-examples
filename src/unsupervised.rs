use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// K-Means
use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
// DBSCAN
use smartcore::cluster::dbscan::{DBSCANParameters, DBSCAN};
// PCA
use smartcore::decomposition::pca::{PCAParameters, PCA};
use smartcore::metrics::*;
// SVD
use smartcore::decomposition::svd::{SVDParameters, SVD};
use smartcore::linalg::svd::SVDDecomposableMatrix;
use smartcore::linalg::BaseMatrix;

use crate::utils;

pub fn digits_clusters() {
    // Load dataset
    let digits_data = digits::load_dataset();
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        digits_data.num_samples,
        digits_data.num_features,
        &digits_data.data,
    );
    // These are our target class labels
    let true_labels = digits_data.target;
    // Fit & predict
    let labels = KMeans::fit(&x, KMeansParameters::default().with_k(10))
        .and_then(|kmeans| kmeans.predict(&x))
        .unwrap();
    // Measure performance
    println!("Homogeneity: {}", homogeneity_score(&true_labels, &labels));
    println!(
        "Completeness: {}",
        completeness_score(&true_labels, &labels)
    );
    println!("V Measure: {}", v_measure_score(&true_labels, &labels));
}

pub fn circles() {
    // Load dataset
    let circles = generator::make_circles(1000, 0.5, 0.05);
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(circles.num_samples, circles.num_features, &circles.data);
    // These are our target class labels
    let true_labels = circles.target;
    // Fit & predict
    let labels = DBSCAN::fit(
        &x,
        DBSCANParameters::default()
            .with_eps(0.2)
            .with_min_samples(5),
    )
    .and_then(|c| c.predict(&x))
    .unwrap();

    // Measure performance
    println!("Homogeneity: {}", homogeneity_score(&true_labels, &labels));
    println!(
        "Completeness: {}",
        completeness_score(&true_labels, &labels)
    );
    println!("V Measure: {}", v_measure_score(&true_labels, &labels));
    utils::scatterplot(
        &x,
        Some(&labels.into_iter().map(|f| f as usize).collect()),
        "test",
    )
    .unwrap();
}

pub fn digits_pca() {
    // Load dataset
    let digits_data = digits::load_dataset();
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        digits_data.num_samples,
        digits_data.num_features,
        &digits_data.data,
    );
    // These are our target class labels
    let labels = digits_data.target;
    // Fit PCA to digits dataset
    let pca = PCA::fit(&x, PCAParameters::default().with_n_components(2)).unwrap();
    // Reduce dimensionality of X
    let x_transformed = pca.transform(&x).unwrap();
    // Plot transformed X to 2 principal components
    utils::scatterplot(
        &x_transformed,
        Some(&labels.into_iter().map(|f| f as usize).collect()),
        "digits_pca",
    )
    .unwrap();
}

pub fn digits_svd1() {
    // Load dataset
    let digits_data = digits::load_dataset();
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        digits_data.num_samples,
        digits_data.num_features,
        &digits_data.data,
    );
    // These are our target class labels
    let labels = digits_data.target;
    // Fit SVD to digits dataset
    let svd = SVD::fit(&x, SVDParameters::default().with_n_components(2)).unwrap();
    // Reduce dimensionality of X
    let x_transformed = svd.transform(&x).unwrap();
    // Plot transformed X to 2 principal components
    utils::scatterplot(
        &x_transformed,
        Some(&labels.into_iter().map(|f| f as usize).collect()),
        "digits_svd",
    )
    .unwrap();
}

pub fn digits_svd2() {
    // Load dataset
    let digits_data = digits::load_dataset();
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        digits_data.num_samples,
        digits_data.num_features,
        &digits_data.data,
    );
    // Decompose matrix into U . Sigma . V^T
    let svd = x.svd().unwrap();
    let u: &DenseMatrix<f32> = &svd.U; //U
    let v: &DenseMatrix<f32> = &svd.V; // V
    let s: &DenseMatrix<f32> = &svd.S(); // Sigma
                                         // Print dimensions of components
    println!("U is {}x{}", u.shape().0, u.shape().1);
    println!("V is {}x{}", v.shape().0, v.shape().1);
    println!("sigma is {}x{}", s.shape().0, s.shape().1);
    // Restore original matrix
    let x_hat = u.matmul(s).matmul(&v.transpose());
    for (x_i, x_hat_i) in x.iter().zip(x_hat.iter()) {
        assert!((x_i - x_hat_i).abs() < 1e-3)
    }
}
