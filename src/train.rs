use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::*;
use burn::optim::*;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::vision::MnistDataset;
use burn::record::CompactRecorder;
use burn::train::metric::LossMetric;
use burn::nn::loss::MseLoss;
use burn::nn::loss::Reduction;
use burn::train::metric::{Adaptor, LossInput};

use crate::data::{MnistBatch, MnistBatcher};
use crate::model::{VAE, ModelConfig};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

/// ディレクトリの初期化関数
fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub struct VAEOutput<B: Backend> {
    pub loss: Tensor<B, 0>,
}

impl<B: Backend> Adaptor<LossInput<B>> for VAEOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone().unsqueeze())
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, VAEOutput<B>> for VAE<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<VAEOutput<B>> {
        let loss = self.forward_loss(batch.images.clone(), &batch.images.device());
        TrainOutput::new(self, loss.backward(), VAEOutput { loss })
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, VAEOutput<B>> for VAE<B> {
    fn step(&self, batch: MnistBatch<B>) -> VAEOutput<B> {
        let loss = self.forward_loss(batch.images.clone(), &batch.images.device());
        VAEOutput { loss }
    }
}

impl<B: Backend> VAE<B> {
    pub fn forward_loss(&self, images: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 0> {
        let (recon_images, mu, logvar) = self.forward(images.clone(), device);

        // 再構成損失（MSE）
        let recon_loss = MseLoss::new().forward(recon_images, images, Reduction::Mean);

        // KLダイバージェンス
        let mu_squared = mu.powf(Tensor::<B, 2>::from_data(TensorData::from([2.0]), device));
        let one = Tensor::<B, 2>::from_data(TensorData::from([1.0]), device);
        let kl_elements = one + logvar.clone() - mu_squared - logvar.exp();

        // `flatten()` 呼び出し時にランク `2` を明示
        let kl_loss = kl_elements.clone().flatten::<2>(0, kl_elements.shape().dims::<2>().len() - 1)
            .mean()
            .mul_scalar(-0.5);

        // 合計損失
        let recon_loss_zero = recon_loss.squeeze(0);
        let kl_loss_zero = kl_loss.squeeze(0);
        recon_loss_zero + kl_loss_zero
    }
}


/// トレーニング関数
pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    // シードの設定
    B::seed(config.seed);

    // データローダーの設定
    let batcher_train = MnistBatcher::<B>::new(device.clone());
    let batcher_valid = MnistBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    // モデルの初期化 (from_config 使用)
    let model = VAE::from_config(&config.model, &device);

    // 学習器の設定
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            model,
            config.optimizer.init(),
            config.learning_rate,
        );

    // モデルの訓練
    let model_trained = learner.fit(dataloader_train, dataloader_test);

    // モデルの保存
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model is saved successfully");
}
