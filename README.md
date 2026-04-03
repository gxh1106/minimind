# Gated Attention Mechanism for MiniMind

This repository contains an enhanced version of the **MiniMind** architecture, featuring an integrated **Gated Attention Mechanism**. 

## Key Modifications

The standard `Attention` module has been modified to incorporate a gating mechanism directly into the query projection and the final output stage. We support two types of gating mechanisms:

1.  **Head-wise Gating (`headwise_attn_output_gate`)**: Applies a scalar gate to each attention head.
2.  **Element-wise Gating (`elementwise_attn_output_gate`)**: Applies a gated transformation to each dimension of the attention output, providing finer granularity (default).

### Architectural Changes
*   **Modified Projection Layers**: The `q_proj` layer has been expanded to output additional gating scores alongside the standard Query projections.
*   **Gated Forward Pass**: In the `forward` pass, these scores are extracted, passed through a `sigmoid` activation function, and multiplied with the attention output before the final projection.
*   **Compatibility**: Maintains compatibility with Flash Attention and Key-Value (KV) caching.

## How to Train

To enable the Gated Attention mechanism during training, use the `--attn_gate` flag.

```bash
# Example training command
python train_XXX.py --attn_gate 1 [other_training_args]
```

*   **Note: `--attn_gate`**: Configure the gated attention mechanism. `0` (default) disables gating. `1` applies **Element-wise Gating**, while `2` applies **Head-wise Gating**.
*   Ensure your `MiniMindConfig` is updated to handle the new gating boolean flags (`headwise_attn_output_gate` and `elementwise_attn_output_gate`) to match the logic implemented in the `Attention` class.

## 🚀 Pre-trained & Fine-tuned Models

We have released multiple versions of MiniMind checkpoints, including pre-trained models, SFT models, and so on. You can find the checkpoints at the following Hugging Face repository:

### MiniMind3 (Latest)
Our latest models equipped with the **Gated Attention** mechanism. 

**[MiniMind3-GatedAttn](https://huggingface.co/XinghaoGuo/MiniMind3-GatedAttn)**

### MiniMind2 (Previous)
Previous models based on the `minimind2` codebase.

**[MiniMind2-GatedAttn](https://huggingface.co/XinghaoGuo/MiniMind2-GatedAttn)**

You can easily load this model using the standard Hugging Face `transformers` library.

## Contributing
This implementation is based on the [MiniMind](https://github.com/jingyaogong/minimind) framework. We encourage users to experiment with different gating configurations to further optimize performance on specific downstream tasks.