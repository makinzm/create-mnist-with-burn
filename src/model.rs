use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Linear, LinearConfig,
    },
    tensor::activation::{relu, sigmoid},
    prelude::*,
    tensor::TensorData,
};
use nn::PaddingConfig2d;
use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub latent_size: usize,
    pub hidden_size: usize,
    pub conv1_out_channels: usize,
    pub conv2_out_channels: usize,
    pub deconv1_out_channels: usize,
    pub deconv2_out_channels: usize,
    pub linear_input_size: usize,
}

impl ModelConfig {
    /// デフォルト設定を作成
    pub fn default() -> Self {
        Self {
            latent_size: 64,
            hidden_size: 128,
            conv1_out_channels: 16,
            conv2_out_channels: 32,
            deconv1_out_channels: 16,
            deconv2_out_channels: 1,
            linear_input_size: 32 * 4 * 4,
        }
    }
}

/// エンコーダの定義
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    linear_mu: Linear<B>,
    linear_logvar: Linear<B>,
}

impl<B: Backend> Encoder<B> {
    pub fn from_config(config: &ModelConfig, device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([1, config.conv1_out_channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv2: Conv2dConfig::new([config.conv1_out_channels, config.conv2_out_channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            pool: AdaptiveAvgPool2dConfig::new([4, 4]).init(),
            linear_mu: LinearConfig::new(config.linear_input_size, config.latent_size).init(device),
            linear_logvar: LinearConfig::new(config.linear_input_size, config.latent_size).init(device),
        }
    }

    pub fn new(device: &B::Device, _hidden_size: usize, latent_size: usize) -> Self {
        Self {
            conv1: Conv2dConfig::new([1, 16], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv2: Conv2dConfig::new([16, 32], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            pool: AdaptiveAvgPool2dConfig::new([4, 4]).init(),
            linear_mu: LinearConfig::new(32 * 4 * 4, latent_size).init(device),
            linear_logvar: LinearConfig::new(32 * 4 * 4, latent_size).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch_size, height, width] = x.dims();
        let x = x.reshape([batch_size, 1, height, width]);
        let x = relu(self.conv1.forward(x));
        let x = sigmoid(self.conv2.forward(x));
        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, 32 * 4 * 4]);

        let mu = self.linear_mu.forward(x.clone());
        let logvar = self.linear_logvar.forward(x);
        (mu, logvar)
    }
}

/// デコーダの定義
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    linear: Linear<B>,
    deconv1: Conv2d<B>,
    deconv2: Conv2d<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn from_config(config: &ModelConfig, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(config.latent_size, config.linear_input_size).init(device),
            deconv1: Conv2dConfig::new([config.conv2_out_channels, config.deconv1_out_channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            deconv2: Conv2dConfig::new([config.deconv1_out_channels, config.deconv2_out_channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
        }
    }

    pub fn new(device: &B::Device, latent_size: usize) -> Self {
        Self {
            linear: LinearConfig::new(latent_size, 32 * 4 * 4).init(device),
            deconv1: Conv2dConfig::new([32, 16], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            deconv2: Conv2dConfig::new([16, 1], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
        }
    }

    pub fn forward(&self, z: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, _] = z.dims();
        let x = relu(self.linear.forward(z));
        let x = x.reshape([batch_size, 32, 4, 4]);
        let x = relu(self.deconv1.forward(x));
        let x = sigmoid(self.deconv2.forward(x));
        x.reshape([batch_size, 28, 28])
    }
}

/// VAE モデルの定義
#[derive(Module, Debug)]
pub struct VAE<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
}

impl<B: Backend> VAE<B> {
    pub fn from_config(config: &ModelConfig, device: &B::Device) -> Self {
        Self {
            encoder: Encoder::from_config(config, device),
            decoder: Decoder::from_config(config, device),
        }
    }

    pub fn new(device: &B::Device, _hidden_size: usize, latent_size: usize) -> Self {
        Self {
            encoder: Encoder::new(device, _hidden_size, latent_size),
            decoder: Decoder::new(device, latent_size),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, device: &B::Device) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>) {
        let (mu, logvar) = self.encoder.forward(x);
        let std = logvar.clone()
            .mul(Tensor::<B, 2>::from_data(TensorData::from([[0.5f32]]), device))
            .exp();
        let shape = mu.dims();
        let numel = shape.iter().product();
        let eps_data: Vec<f32> = StandardNormal
            .sample_iter(thread_rng())
            .take(numel)
            .collect();
        let eps = Tensor::<B, 2>::from_data(TensorData::from(&eps_data[..]), device).reshape(shape);
        let z = mu.clone().add(std.mul(eps));
        let recon_x = self.decoder.forward(z);
        (recon_x, mu, logvar)
    }
}
