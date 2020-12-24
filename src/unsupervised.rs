use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// K-Means
use smartcore::cluster::kmeans::KMeans;
// PCA
use smartcore::decomposition::pca::PCA;
use smartcore::metrics::*;
// SVD
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
    let labels = KMeans::fit(&x, 10, Default::default())
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
    let pca = PCA::fit(&x, 2, Default::default()).unwrap();
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

pub fn digits_svd() {
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
