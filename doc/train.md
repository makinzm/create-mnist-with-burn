# VAE Training Flow Summary

## トレーニングフローの概要

| Step                  | Description                                         |
|-----------------------|-----------------------------------------------------|
| 1. **Artifact Directory Setup** | 訓練データやモデルの保存先ディレクトリを準備                   |
| 2. **Configuration Initialization** | トレーニング設定 (`TrainingConfig`) を読み込み、シード設定      |
| 3. **Data Loading**             | MNISTデータセットを読み込み、データローダーを構成              |
| 4. **Model Initialization**    | VAEモデルを設定 (`from_config` メソッドを使用)                 |
| 5. **Optimizer Setup**         | Adamオプティマイザを設定、学習率を設定                          |
| 6. **Learner Initialization**  | トレーニングループ (`Learner`) を設定                            |
| 7. **Training Loop**           | 学習データでモデルをトレーニングし、検証データで評価            |
| 8. **Model Saving**            | 訓練後のモデルを保存 (`CompactRecorder`)                         |

---

## 訓練フロー詳細

### 1. Artifact Directory Setup

```rust
create_artifact_dir(artifact_dir);
```

- 既存のディレクトリを削除し、新規に作成します。
- 設定ファイル (`config.json`) を保存。

### 2. Configuration Initialization

```rust
B::seed(config.seed);
```

- シードを設定し、再現性を確保します。

### 3. Data Loading

```rust
let dataloader_train = DataLoaderBuilder::new(batcher_train)
    .batch_size(config.batch_size)
    .shuffle(config.seed)
    .num_workers(config.num_workers)
    .build(MnistDataset::train());
```

- **データセット**:
  - `MnistDataset::train()` および `MnistDataset::test()` を使用。
- **データローダー**:
  - バッチサイズ、シャッフル、ワーカー数を設定。

### 4. Model Initialization

```rust
let model = VAE::from_config(&config.model, &device);
```

- `ModelConfig` に基づいてVAEモデルを初期化します。
- モデルは指定したデバイス（CPU/GPU）に配置されます。

### 5. Optimizer Setup

```rust
config.optimizer.init()
```

- `AdamConfig` からオプティマイザを初期化します。
- 学習率は `config.learning_rate` から設定されます。

### 6. Learner Initialization

```rust
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
```

- **`LearnerBuilder`**:
  - 損失メトリック (`LossMetric`) を設定。
  - チェックポイント (`CompactRecorder`) を設定し、モデルの保存と復元が可能。
  - 使用するデバイス（CPU/GPU）を設定。
  - エポック数 (`num_epochs`) を設定。
  - トレーニングループのサマリーを出力。

### 7. Training Loop

```rust
let model_trained = learner.fit(dataloader_train, dataloader_test);
```

- **`fit` メソッド**:
  - 学習データローダー (`dataloader_train`) と検証データローダー (`dataloader_test`) を使用してトレーニングを行います。
  - エポックごとに学習損失と検証損失が計算され、進行状況が表示されます。

### 8. Model Saving

```rust
model_trained.save_file(format!("{artifact_dir}/model"), &CompactRecorder::new());
```

- 訓練されたモデルを保存します。
- `CompactRecorder` を使用して、軽量な形式で保存。

---

## 損失関数 (`forward_loss`)

```rust
pub fn forward_loss(&self, images: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 0>
```

1. **再構成損失 (Reconstruction Loss)**:
   - MSE (Mean Squared Error) を使用。
   - PyTorchの標準的なVAE実装と同様に、`Reduction::Mean` で平均化。

2. **KLダイバージェンス (KL Divergence Loss)**:
   - 潜在変数の分布と標準正規分布の差異を計算。
   - `KL = 0.5 * Σ (1 + log(σ^2) - μ^2 - σ^2)` で計算。

3. **合計損失 (Total Loss)**:
   - `recon_loss + kl_loss` として計算。

---

## 訓練の全体フロー

```plaintext
1. Config Load → Seed Setting → DataLoader Setup
2. Model Initialization → Optimizer Setup
3. Learner Initialization
4. Training Loop (Learner::fit)
   ├─ Forward Pass (VAE::forward)
   ├─ Loss Calculation (forward_loss)
   ├─ Backpropagation and Optimization
   └─ Evaluation (Validation Data)
5. Model Saving
```

