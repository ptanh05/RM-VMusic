# RM-VMusic Phase 5: Ablation Study Report

## 1. Model Component Ablation Ladder (Models A – E)

| Model Variant | Core Modules Active | IID Accuracy | Macro-F1 (Primary) | Weighted-F1 | Balanced Acc | Gain vs Baseline |
|---------------|---------------------|--------------|--------------------|-------------|--------------|------------------|
| **Model A** | Baseline Standard Concat Fusion | 0.4914 | **0.2584** | 0.5326 | 0.2811 | Reference |
| **Model B** | Dynamic Uncertainty-Aware Reliability Fusion | 0.5284 | **0.2576** | 0.5534 | 0.2775 | -0.0008 |
| **Model C** | Dynamic Reliability + Modality Dropout | 0.4728 | **0.2613** | 0.5170 | 0.2697 | **+0.0029** |
| **Model D** | Dynamic Reliability + Dropout + Distribution Robustness | 0.4728 | **0.2629** | 0.5152 | 0.2697 | **+0.0045 (Peak F1)** |
| **Model E** | Full Proposed (Reliability + Dropout + Robustness + Contrastive) | 0.4704 | **0.2543** | 0.5147 | 0.2622 | -0.0041 |

---

## 2. Modality Dropout & Simulated Missing Modality Ablations (Model E)

| Evaluated Subset Mode | Accuracy | Macro-F1 | Mean Audio Alpha | Mean Lyrics Alpha | Mean Cover Alpha |
|-----------------------|----------|----------|------------------|-------------------|------------------|
| `none` | 0.4704 | **0.2543** | 0.574 | 0.360 | 0.066 |
| `no_audio` | 0.5111 | **0.2339** | 0.062 | 0.764 | 0.175 |
| `no_lyrics` | 0.0951 | **0.0673** | 0.916 | 0.000 | 0.084 |
| `no_cover` | 0.4247 | **0.2465** | 0.621 | 0.378 | 0.000 |
| `no_audio_lyrics` | 0.1062 | **0.0389** | 0.278 | 0.278 | 0.444 |
| `no_audio_cover` | 0.4728 | **0.2327** | 0.081 | 0.837 | 0.081 |
| `no_lyrics_cover` | 0.0333 | **0.0391** | 0.999 | 0.000 | 0.000 |
