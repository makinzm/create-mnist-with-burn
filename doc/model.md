# VAE Model Architecture

### Encoder

| Layer                | Type             | Input Shape           | Output Shape         | Params   |
|----------------------|------------------|-----------------------|----------------------|----------|
| `conv1`              | Conv2d           | `[batch_size, 1, 28, 28]`  | `[batch_size, 16, 14, 14]` | 160      |
| `conv2`              | Conv2d           | `[batch_size, 16, 14, 14]` | `[batch_size, 32, 7, 7]`   | 4,640    |
| `pool`               | AdaptiveAvgPool2d | `[batch_size, 32, 7, 7]`   | `[batch_size, 32, 4, 4]`   | 0        |
| `linear_mu`          | Linear           | `[batch_size, 32 * 4 * 4]` | `[batch_size, latent_size]` | \(32 * 4 * 4 * \text{latent_size}\) |
| `linear_logvar`      | Linear           | `[batch_size, 32 * 4 * 4]` | `[batch_size, latent_size]` | \(32 * 4 * 4 * \text{latent_size}\) |

### Decoder

| Layer                | Type             | Input Shape             | Output Shape          | Params   |
|----------------------|------------------|-------------------------|-----------------------|----------|
| `linear`             | Linear           | `[batch_size, latent_size]` | `[batch_size, 32 * 4 * 4]` | \(32 * 4 * 4 * \text{latent_size}\) |
| `deconv1`            | Conv2d           | `[batch_size, 32, 4, 4]`   | `[batch_size, 16, 7, 7]`   | 4,640    |
| `deconv2`            | Conv2d           | `[batch_size, 16, 7, 7]`   | `[batch_size, 1, 14, 14]`  | 145      |
| Reshape              | Reshape          | `[batch_size, 1, 14, 14]`  | `[batch_size, 28, 28]`     | 0        |

### VAE Summary

| Module   | Input Shape               | Output Shape              |
|----------|---------------------------|---------------------------|
| Encoder  | `[batch_size, 28, 28]`    | `(mu, logvar): [batch_size, latent_size]` |
| Sampling | `(mu, logvar)`            | `[batch_size, latent_size]` |
| Decoder  | `[batch_size, latent_size]` | `[batch_size, 28, 28]`    |

---

### パラメータの合計数

- `conv1`：160個
- `conv2`：4,640個
- `linear_mu` と `linear_logvar`：\(32 * 4 * 4 * \text{latent_size}\) 個ずつ
- `linear`：\(32 * 4 * 4 * \text{latent_size}\) 個
- `deconv1`：4,640個
- `deconv2`：145個

