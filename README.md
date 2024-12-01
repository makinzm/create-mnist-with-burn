# create-mnist-with-burn


## SetUp

```shell
cargo build

# Train
cargo run -- --mode train
```

## ERROR 1

```shell

     Running `target/debug/create-mnist-with-burn --mode train`
Training mode selected.
There is no model file. Start training.
Initializing model...
Building learner...
=== PANIC ===
A fatal error happened, you can check the experiment logs here => '~/**/create-mnist-with-burn/artifact/experiment.log'
=============
thread 'main' panicked at ~/.cargo/registry/src/index.crates.io-6f+++/burn-tensor-0.15.0/src/tensor/api/base.rs:722:9:
=== Tensor Operation Error ===
  Operation: 'From Data'
  Reason:
    1. Given dimensions differ from the tensor rank. Tensor rank: '2', given dimensions: '[1]'.

stack backtrace:
   0: rust_begin_unwind
             at /rustc/f6+++/library/std/src/panicking.rs:662:5
   1: core::panicking::panic_fmt
             at /rustc/f6+++/library/core/src/panicking.rs:74:14
   2: core::panicking::panic_display
             at /rustc/f6+++/library/core/src/panicking.rs:264:5
   3: burn_tensor::tensor::api::base::Tensor<B,_,K>::from_data::panic_cold_display
             at /rustc/f6+++/library/core/src/panic.rs:100:13
   4: burn_tensor::tensor::api::base::Tensor<B,_,K>::from_data
             at ~/.cargo/registry/src/index.crates.io-6f+++/burn-tensor-0.15.0/src/tensor/api/check.rs:1181:17
   5: create_mnist_with_burn::model::VAE<B>::forward
             at ./src/model.rs:171:18
   6: create_mnist_with_burn::train::<impl create_mnist_with_burn::model::VAE<B>>::forward_loss
             at ./src/train.rs:64:42
   7: create_mnist_with_burn::train::<impl burn_train::learner::train_val::TrainStep<create_mnist_with_burn::data::MnistBatch<B>,create_mnist_with_burn::train::VAEOutput<B>> for create_mnist_with_burn::model::VAE<B>>::step
             at ./src/train.rs:50:20
   8: burn_train::learner::epoch::TrainEpoch<TI>::run
             at ~/.cargo/registry/src/index.crates.io-6f+++/burn-train-0.15.0/src/learner/epoch.rs:113:24
   9: burn_train::learner::train_val::<impl burn_train::learner::base::Learner<LC>>::fit
             at ~/.cargo/registry/src/index.crates.io-6f+++/burn-train-0.15.0/src/learner/train_val.rs:165:44
  10: create_mnist_with_burn::train::train
             at ./src/train.rs:136:25
  11: create_mnist_with_burn::call_train
             at ./src/main.rs:32:9
  12: create_mnist_with_burn::main::{{closure}}
             at ./src/main.rs:49:17
  13: std::panicking::try::do_call
             at /rustc/f6+++/library/std/src/panicking.rs:554:40
  14: __rust_try
  15: std::panicking::try
             at /rustc/f6+++/library/std/src/panicking.rs:518:19
  16: std::panic::catch_unwind
             at /rustc/f6+++/library/std/src/panic.rs:345:14
  17: create_mnist_with_burn::main
             at ./src/main.rs:48:26
  18: core::ops::function::FnOnce::call_once
             at /rustc/f6+++/library/core/src/ops/function.rs:250:5
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
An error occurred during training.
Caught panic: Err(Any { .. })
```

## ERROR 2

- Related to [Compilation fails with `'Error in Surface::configure: parent device is lost'` when Nvidia GPU is selected on Linux · Issue #2519 · gfx-rs/wgpu](https://github.com/gfx-rs/wgpu/issues/2519)

```shell
> cargo run -- --mode train
...
...

=== Tensor Operation Error ===
  Operation: 'From Data'
  Reason:
    1. Given dimensions differ from the tensor rank. Tensor rank: '2', given dimensions: '[1]'.

note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
An error occurred during training.
Caught panic: Err(Any { .. })
Memory block wasn't deallocated
=== PANIC ===

...
...

wgpu error: Validation Error

Caused by:
  In Queue::write_buffer_with
    Parent device is lost

...
...
```
