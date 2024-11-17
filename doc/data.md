# MnistBatcher Data Summary

### `MnistBatcher` クラスの概要

| Attribute   | Type        | Description                         |
|-------------|-------------|-------------------------------------|
| `device`    | `B::Device` | データを配置するデバイス（CPU/GPU） |

### `MnistBatch` クラスの概要

| Attribute   | Type               | Shape                      | Description                   |
|-------------|--------------------|----------------------------|-------------------------------|
| `images`    | `Tensor<B, 3>`     | `[batch_size, 1, 28, 28]`  | バッチ化されたMNIST画像       |
| `targets`   | `Tensor<B, 1, Int>`| `[batch_size]`             | バッチ化されたラベル          |

---

## `MnistBatcher` 処理フロー概要

| Step             | Operation                             | Input Shape       | Output Shape      | Description                                          |
|------------------|---------------------------------------|-------------------|-------------------|------------------------------------------------------|
| **1. Image Load**| テンソルデータへの変換               | `[28, 28]`        | `[1, 28, 28]`     | 画像を2次元テンソルに変換し、リシェイプする          |
| **2. Normalize** | ピクセル値の標準化                   | `[1, 28, 28]`     | `[1, 28, 28]`     | 平均0.1307、標準偏差0.3081で標準化                   |
| **3. Label Load**| ラベルをテンソルに変換               | `int`             | `[1]`             | ラベルを `Int` 型のテンソルに変換                    |
| **4. Batch Cat** | テンソルの結合（バッチ化）           | `Vec<Tensor>`     | `[batch_size, 1, 28, 28]` | 全ての画像とラベルを結合してバッチ化                 |
| **5. To Device** | データを指定デバイスに移動           | `[batch_size, 1, 28, 28]` | `[batch_size, 1, 28, 28]` | バッチ化されたデータをCPU/GPUに配置                  |

---

## 全体のデータフロー

```plaintext
MnistItem (画像, ラベル)
  ↓
MnistBatcher::batch()
  ↓
[28, 28] → [1, 28, 28] (リシェイプ)
  ↓
標準化: (image / 255 - 0.1307) / 0.3081
  ↓
ラベル: int → Tensor<Int>
  ↓
バッチ化: Tensorを結合
  ↓
デバイスに配置 (CPU/GPU)
  ↓
MnistBatch { images, targets }
```

---

### Summary

- **Normalization**:
  - 画像データは、MNISTの標準的な平均と標準偏差（PyTorchのMNISTサンプルに基づく）で標準化されます。
- **Batching**:
  - `batch()` メソッドにより、複数のデータ項目 (`MnistItem`) がバッチ (`MnistBatch`) にまとめられます。
- **Device Placement**:
  - バッチ化されたデータは、計算デバイス（CPU/GPU）に配置され、効率的な処理が可能になります。


