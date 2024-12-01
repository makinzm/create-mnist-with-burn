# create-mnist-with-burn


## SetUp

```shell
cargo build

# Train
cargo run -- --mode train
```

## ERROR 1

```shell
/.cargo/registry/src/index.crates.io-***/burn-tensor-0.15.0/src/tensor/api/base.rs:722:9:
=== Tensor Operation Error ===
  Operation: 'From Data'
  Reason:
    1. Given dimensions differ from the tensor rank. Tensor rank: '2', given dimensions: '[1]'.

note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
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
