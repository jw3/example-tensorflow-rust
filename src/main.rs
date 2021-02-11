use std::error::Error;

use tensorflow::DataType;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::Shape;
use tensorflow::Tensor;
use tensorflow::{Graph, SessionRunArgs};

// modified from
// https://github.com/tensorflow/rust/blob/c1a1f6bb0cea568ac301d7c58acfab797df18d63/src/lib.rs#L2210
fn main() -> Result<(), Box<dyn Error>> {
    let mut g = Graph::new();

    let x_op = {
        let mut nd = g.new_operation("Placeholder", "x").unwrap();
        nd.set_attr_type("dtype", DataType::String).unwrap();
        nd.set_attr_shape("shape", &Shape::new(Some(vec![])))
            .unwrap();
        nd.finish().unwrap()
    };
    let y_op = {
        let mut nd = g.new_operation("Print", "y").unwrap();
        nd.add_input(x_op.clone());
        nd.add_input_list(vec![x_op.clone().into()].as_slice());
        nd.finish().unwrap()
    };

    let options = SessionOptions::new();
    let session = Session::new(&options, &g).unwrap();
    let mut x = <Tensor<String>>::new(&[2]);
    x[0] = "foo".to_string();
    x[1] = "bar".to_string();

    let mut step = SessionRunArgs::new();
    step.add_feed(&x_op, 0, &x);
    let output_ix = step.request_fetch(&y_op, 0);
    session.run(&mut step).unwrap();
    let output_tensor = step.fetch::<String>(output_ix).unwrap();

    assert_eq!(output_tensor.len(), 2);
    assert_eq!(output_tensor[0], "foo");
    assert_eq!(output_tensor[1], "bar");

    Ok(())
}
