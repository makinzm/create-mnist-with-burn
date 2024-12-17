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
    pub loss: Tensor<B, 1>,
}

impl<B: Backend> Adaptor<LossInput<B>> for VAEOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone().unsqueeze())
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, VAEOutput<B>> for VAE<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<VAEOutput<B>> {
        let loss = self.forward_loss(batch.images.clone(), &batch.images.device());
        println!("Loss shape: {:?}", loss.dims());
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
    pub fn forward_loss(&self, images: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 1> {
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
        let recon_loss_zero = recon_loss;
        let kl_loss_zero = kl_loss;
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
    println!("Initializing model...");
    let model = VAE::from_config(&config.model, &device);
    println!("Conv1 weight shape: {:?}", model.encoder().conv1().weight.dims());
    println!("Conv1 bias shape: {:?}", model.encoder().conv1().bias.as_ref().map(|b| b.dims()));
    println!("Conv2 weight shape: {:?}", model.encoder().conv2().weight.dims());
    println!("Conv2 bias shape: {:?}", model.encoder().conv2().bias.as_ref().map(|b| b.dims()));
    println!("Linear_mu weight shape: {:?}", model.encoder().linear_mu().weight.dims());
    println!("Linear_mu bias shape: {:?}", model.encoder().linear_mu().bias.as_ref().map(|b| b.dims()));
    println!("Linear_logvar weight shape: {:?}", model.encoder().linear_logvar().weight.dims());
    println!("Linear_logvar bias shape: {:?}", model.encoder().linear_logvar().bias.as_ref().map(|b| b.dims()));

    println!("Linear weight shape: {:?}", model.decoder().linear().weight.dims());
    println!("Linear bias shape: {:?}", model.decoder().linear().bias.as_ref().map(|b| b.dims()));
    println!("Deconv1 weight shape: {:?}", model.decoder().deconv1().weight.dims());
    println!("Deconv1 bias shape: {:?}", model.decoder().deconv1().bias.as_ref().map(|b| b.dims()));
    println!("Deconv2 weight shape: {:?}", model.decoder().deconv2().weight.dims());
    println!("Deconv2 bias shape: {:?}", model.decoder().deconv2().bias.as_ref().map(|b| b.dims()));

    println!("Learning rate: {:?}", config.learning_rate);
    let optimizer = config.optimizer.init();

    let batcher_train = MnistBatcher::<B>::new(device.clone());
    // Debug
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(1) // 1件だけ取得する
        .shuffle(config.seed)
        .build(MnistDataset::train());

    for (i, batch) in dataloader_train.iter().take(1).enumerate() {
        println!("Batch {} images shape: {:?}", i, batch.images.dims());
        println!("Batch {} targets shape: {:?}", i, batch.targets.dims());
    }

    // 学習器の設定
    println!("Building learner...");

    let learner = LearnerBuilder::new(artifact_dir);
    println!("LearnerBuilder initialized.");
    let learner = learner
        .metric_train_numeric(LossMetric::new());
    println!("LearnerBuilder metric_train_numeric set.");
    let learner = learner
        .metric_valid_numeric(LossMetric::new());
    println!("LearnerBuilder metric_valid_numeric set.");
    let learner = learner
        .with_file_checkpointer(CompactRecorder::new());
    println!("LearnerBuilder with_file_checkpointer set.");
    let learner = learner
        .devices(vec![device.clone()]);
    println!("LearnerBuilder devices set.");
    let learner = learner
        .num_epochs(config.num_epochs);
    println!("LearnerBuilder num_epochs set.");
    let learner = learner
        .build(
            model,
            optimizer,
            config.learning_rate,
        );
    println!("LearnerBuilder build done.");
    // TODO: The below is not reached.

    println!("Start training...");
    // モデルの訓練
    let model_trained = learner.fit(dataloader_train, dataloader_test);

    // モデルの保存
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model is saved successfully");
}

