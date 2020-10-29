// plotters
use plotters::prelude::*;
// DenseMatrix wrapper around Vec
use smartcore::math::num::RealNumber;
use smartcore::linalg::BaseMatrix;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;

/// Get min value of `x` along axis `axis`
pub fn min<T: RealNumber>(x: &DenseMatrix<T>, axis: usize) -> T {
    let n = x.shape().0;
    x.slice(0..n, axis..axis+1).iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
}

/// Get max value of `x` along axis `axis`
pub fn max<T: RealNumber>(x: &DenseMatrix<T>, axis: usize) -> T {
    let n = x.shape().0;
    x.slice(0..n, axis..axis+1).iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
}

/// Draw a mesh grid defined by `mesh` with a scatterplot of `data` on top
/// We use Plotters library to draw scatter plot.
/// https://docs.rs/plotters/0.3.0/plotters/
pub fn scatterplot_with_mesh(
    mesh: &DenseMatrix<f32>,
    mesh_labels: &Vec<f32>,
    data: &DenseMatrix<f32>,
    labels: &Vec<f32>,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}.svg", title);
    let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();  
    
    root.fill(&WHITE)?;
    let root = root.margin(15, 15, 15, 15);

    let x_min = (min(mesh, 0) - 1.0) as f64;
    let x_max = (max(mesh, 0) + 1.0) as f64;
    let y_min = (min(mesh, 1) - 1.0) as f64;
    let y_max = (max(mesh, 1) + 1.0) as f64;

    let mesh_labels: Vec<usize> = mesh_labels.into_iter().map(|&v| v as usize).collect();
    let mesh: Vec<f64> = mesh.iter().map(|v| v as f64).collect(); 
    
    let labels: Vec<usize> = labels.into_iter().map(|&v| v as usize).collect();
    let data: Vec<f64> = data.iter().map(|v| v as f64).collect(); 

    let mut scatter_ctx  = ChartBuilder::on(&root)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;
    scatter_ctx.draw_series(mesh.chunks(2).zip(mesh_labels.iter()).map(|(xy, &l)| {
        EmptyElement::at((xy[0], xy[1]))
            + Circle::new((0, 0), 1, ShapeStyle::from(&Palette99::pick(l)).filled())
    }))?;
    scatter_ctx.draw_series(data.chunks(2).zip(labels.iter()).map(|(xy, &l)| {
        EmptyElement::at((xy[0], xy[1]))
            + Circle::new((0, 0), 3, ShapeStyle::from(&Palette99::pick(l + 3)).filled())
    }))?;

    Ok(())
}

/// Draw a scatterplot of `data` with labels `labels`
/// We use Plotters library to draw scatter plot.
/// https://docs.rs/plotters/0.3.0/plotters/
pub fn scatterplot(
    data: &DenseMatrix<f32>,
    labels: &Vec<f32>,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}.svg", title);
    let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();
    
    let x_min = (min(data, 0) - 1.0) as f64;
    let x_max = (max(data, 0) + 1.0) as f64;
    let y_min = (min(data, 1) - 1.0) as f64;
    let y_max = (max(data, 1) + 1.0) as f64;
    
    root.fill(&WHITE)?;
    let root = root.margin(10, 10, 10, 10);

    let labels: Vec<usize> = labels.into_iter().map(|&v| v as usize).collect();
    let data_values: Vec<f64> = data.iter().map(|v| v as f64).collect();    

    let mut scatter_ctx = ChartBuilder::on(&root)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
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

/// Generates 2x2 mesh grid from `x`
pub fn make_meshgrid(x: &DenseMatrix<f32>) -> DenseMatrix<f32> {
    let n = x.shape().0;
    let x_min = min(x, 0) - 1.0;
    let x_max = max(x, 0) + 1.0;
    let y_min = min(x, 1) - 1.0;
    let y_max = max(x, 1) + 1.0;

    let x_step = (x_max - x_min) / n as f32;
    let x_axis: Vec<f32> = (0..n).map(|v| (v as f32 * x_step) + x_min).collect();            
    let y_step = (y_max - y_min) / n as f32;
    let y_axis: Vec<f32> = (0..n).map(|v| (v as f32 * y_step) + y_min).collect();            
    
    let x_new: Vec<Vec<f32>> = x_axis.clone().into_iter().flat_map(move |v1| {        
        y_axis.clone().into_iter().map(move |v2| vec!(v1, v2))
    }).collect();

    DenseMatrix::from_2d_vec(&x_new)
}