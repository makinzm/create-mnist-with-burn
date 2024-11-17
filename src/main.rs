mod model;
mod data;
mod train;

use clap::Parser;
use std::panic;

use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};

use crate::train::TrainingConfig;
use crate::model::ModelConfig;

#[derive(Parser, Debug)]
#[command(author = "Your Name", version = "1.0", about = "VAE Training and Inference")]
struct Args {
    #[arg(short, long, default_value = "train")]
    mode: String,
}

fn call_train() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let current_dir = std::env::current_dir().unwrap();
    let artifact_dir = current_dir.join("artifact");
    // if !Path::new(&artifact_dir).exists() {
        println!("There is no model file. Start training.");
        crate::train::train::<MyAutodiffBackend>(
            artifact_dir.to_str().unwrap(),
            TrainingConfig::new(ModelConfig::default(), AdamConfig::new()),
            device,
        )
    // } else {
    //     println!("Model file exists.");
    // }
}

fn main() {
    let args = Args::parse();

    match args.mode.as_str() {
        "train" => {
            println!("Training mode selected.");
            let result = panic::catch_unwind(|| {
                call_train();
            });
            if result.is_err() {
                eprintln!("An error occurred during training.");
                println!("Caught panic: {:?}", result);
            }
        }
        "infer" => {
            println!("Inference mode selected.");
        }
        _ => {
            eprintln!("Invalid mode. Please choose 'train' or 'infer'.");
        }
    }
}

