# Next Steps for Improvement

## Current Status (Rank #14, Score: 0.861)

**Our Performance:**
- ROI MSE: 0.025135
- Full MSE: 0.005383
- Model: Attention, LD=16, 15.27MB

**Top Performers:**
- **Rank #1** (0.959): ROI=0.000038 (660x better), LD=16, 13.21MB
- **Rank #2** (0.946): ROI=0.002805 (9x better), LD=16, 15.53MB
- **Rank #3** (0.944): ROI=0.004237, LD=8, 21.47MB

**Critical Gap**: ROI reconstruction quality is our bottleneck.

## What We've Tried

1. ✅ Attention mechanisms (channel attention)
2. ✅ Combined loss (perceptual + ROI-weighted MSE)
3. ✅ Extreme ROI weighting (lambda=50, roi_size=0.10)
4. ✅ Data augmentation (strong)
5. ❌ Skip connections (TorchScript incompatible)
6. ❌ Advanced model with CBAM (TorchScript incompatible)

## What to Explore Next

**Architecture Ideas:**
1. **Fix TorchScript + skip connections** - U-Net is critical for reconstruction
2. Try **LD=8** with maxed-out model size (21-22MB) like rank #3
3. Research **VAE** or **diffusion-based** autoencoders
4. Implement **multi-scale loss** (losses at different resolutions)
5. Try **transformer-based** bottleneck instead of FC layer

**Training Tricks:**
- **Search the web** for "traffic light detection neural networks" and "autoencoder tricks 2024"
- Look into **focal loss** for ROI regions
- Try **curriculum learning** (coarse→fine)
- Investigate **self-supervised pretraining**

**Critical**: Top performers likely have skip connections working. Solve TorchScript compatibility or find alternatives!

