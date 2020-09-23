use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// K-Means
use smartcore::cluster::kmeans::KMeans;
// PCA
use smartcore::decomposition::pca::PCA;
use smartcore::metrics::*;
// plotters
use plotters::prelude::*;

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

    let kmeans = KMeans::fit(&x, 10, Default::default()).unwrap();

    let labels = kmeans.predict(&x).unwrap();

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

    let pca = PCA::fit(&x, 2, Default::default()).unwrap();

    let x_transformed = pca.transform(&x).unwrap();

    scatterplot(&x_transformed, &labels, "digits_pca").unwrap();
}

fn scatterplot(
    data: &DenseMatrix<f32>,
    labels: &Vec<f32>,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}.svg", title);
    let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;
    let root = root.margin(10, 10, 10, 10);

    let labels: Vec<usize> = labels.into_iter().map(|&v| v as usize).collect();
    let data_values: Vec<f64> = data.iter().map(|v| v as f64).collect();

    let mut scatter_ctx = ChartBuilder::on(&root)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(-40f64..40f64, -40f64..40f64)?;
    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;
    scatter_ctx.draw_series(data_values.chunks(2).zip(labels.iter()).map(|(xy, &l)| {
        EmptyElement::at((xy[0], xy[1]))
            + Circle::new((0, 0), 3, ShapeStyle::from(&Palette99::pick(l)).filled())
            + Text::new(format!("{}", l), (6, 0), ("sans-serif", 15.0).into_font())
    }))?;

    Ok(())
}
