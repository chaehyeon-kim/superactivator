# Key Percentage-to-Layer Mappings

## 81% and 92% Layer Mappings

| Model | Total Layers | 81% → Layer | 92% → Layer |
|-------|--------------|-------------|-------------|
| CLIP | 24 | Layer 18 | Layer 21 |
| Llama-Vision | 40 | Layer 31 | Layer 35 |
| Gemma-2 (SAE) | 42 | Layer 33 | Layer 37 |
| Llama-Text | 32 | Layer 24 | Layer 28 |
| Gemma-Text | 28 | Layer 21 | Layer 24 |
| Qwen-Text | 32 | Layer 24 | Layer 28 |

## Percentage Ranges per Layer

### CLIP (24 layers)
- **Layer 18**: 80-83% (contains 81%)
- **Layer 21**: 92-95% (contains 92%)

### Llama-Vision (40 layers)
- **Layer 31**: 80-82% (contains 81%)
- **Layer 35**: 90-92% (contains 92%)

### Gemma-2 SAE (42 layers)
- **Layer 33**: 81-83% (contains 81%)
- **Layer 37**: 91-92% (contains 92%)

### Text Models
- **Llama-Text & Qwen-Text** (32 layers):
  - Layer 24: 79-81% (contains 81%)
  - Layer 28: 91-93% (contains 92%)
- **Gemma-Text** (28 layers):
  - Layer 21: 79-82% (contains 81%)
  - Layer 24: 90-92% (contains 92%)

## Quick Reference
The percentage ranges show that each layer typically covers 2-4% of the model depth, with slight variations based on the total number of layers.