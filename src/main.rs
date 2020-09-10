pub mod quick_start;

use std::collections::HashMap;
use structopt::StructOpt;

#[derive(StructOpt)]
/// Run SmartCore example
struct Cli {
    /// The example to run. Pass `list_examples` to list all available examples!    
    example_name: String
}

fn main() {

    let args = Cli::from_args();

    let examples = args.example_name;

    let all_examples: HashMap<&str, &dyn Fn()> = vec![        
        ("quick_start:iris-knn", &quick_start::iris_knn_example as &dyn Fn()),
        ("quick_start:iris-lr", &quick_start::iris_lr_example as &dyn Fn()),
        ("quick_start:iris-lr-ndarray", &quick_start::iris_lr_ndarray_example as &dyn Fn())
    ].into_iter().collect();

    match examples {
        example if all_examples.contains_key(&example.as_str()) => {
            println!("Running {} ...\n", example);
            for example_fn in all_examples.get(example.as_str()) {
                example_fn();
            }
            println!("\nDone!");
            
        },
        example if example == "list_examples" || example == "list" => {
            println!("I can run following examples:");
            for c in all_examples.keys() {
                println!("\t{}", c);
            }
        },
        example => eprintln!("Can't find this example: [{}]. Type `list` to list all available examples", example)    
    }
}
