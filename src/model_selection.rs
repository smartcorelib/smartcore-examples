use smartcore::dataset::iris::load_dataset;
use std::fs::File;
use std::io::prelude::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Imports for KNN classifier
use smartcore::math::distance::*;
use smartcore::neighbors::knn_classifier::*;
// Model performance
use smartcore::metrics::accuracy;

pub fn save_restore_knn() {
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
    let knn = KNNClassifier::fit(
        &x,
        &y,
        Distances::euclidian(), // We use euclidian distance here.
        Default::default(),
    )
    .unwrap();
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
