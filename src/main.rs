pub mod model_selection;
pub mod quick_start;
pub mod supervised;
pub mod unsupervised;
pub mod utils;

use std::collections::HashMap;
use structopt::StructOpt;

#[derive(StructOpt)]
/// Run SmartCore example
struct Cli {
    /// The example to run. Pass `list_examples` to list all available examples!    
    example_name: String,
}

fn main() {
    let args = Cli::from_args();

    let examples = args.example_name;

    let all_examples: HashMap<&str, &dyn Fn()> = vec![
        (
            "quick-start:iris-knn",
            &quick_start::iris_knn_example as &dyn Fn(),
        ),
        (
            "quick-start:iris-lr",
            &quick_start::iris_lr_example as &dyn Fn(),
        ),
        (
            "quick-start:iris-lr-ndarray",
            &quick_start::iris_lr_ndarray_example as &dyn Fn(),
        ),
        (
            "quick-start:iris-lr-nalgebra",
            &quick_start::iris_lr_nalgebra_example as &dyn Fn(),
        ),
        (
            "quick-start:iris-gaussiannb",
            &quick_start::iris_gaussiannb_example as &dyn Fn(),
        ),
        (
            "supervised:breast-cancer",
            &supervised::breast_cancer as &dyn Fn(),
        ),
        ("supervised:boston", &supervised::boston as &dyn Fn()),
        (
            "unsupervised:digits_clusters",
            &unsupervised::digits_clusters as &dyn Fn(),
        ),
        (
            "unsupervised:digits_pca",
            &unsupervised::digits_pca as &dyn Fn(),
        ),
        (
            "unsupervised:digits_svd",
            &unsupervised::digits_svd as &dyn Fn(),
        ),
        (
            "model_selection:save_restore_knn",
            &model_selection::save_restore_knn as &dyn Fn(),
        ),
        ("supervised:svm", &supervised::svm as &dyn Fn()),
    ]
    .into_iter()
    .collect();

    match examples {
        example if all_examples.contains_key(&example.as_str()) => {
            println!("Running {} ...\n", example);
            for example_fn in all_examples.get(example.as_str()) {
                example_fn();
            }
            println!("\nDone!");
        }
        example if example == "list_examples" || example == "list" => {
            println!("You can run following examples:");
            let mut keys: Vec<&&str> = all_examples.keys().collect();
            keys.sort();
            for c in keys {
                println!("\t{}", c);
            }
        }
        example => eprintln!(
            "Can't find this example: [{}]. Type `list` to list all available examples",
            example
        ),
    }
}
