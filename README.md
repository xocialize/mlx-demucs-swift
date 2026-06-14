# mlx-demucs-swift

The MLXEngine **`audioSeparation`** package over [HTDemucs v4](https://github.com/facebookresearch/demucs) — vocal source separation on Apple Silicon.

A thin conformance layer that wraps the standalone inference engine
[`demucs-mlx-swift`](https://github.com/xocialize/demucs-mlx-swift) (product `SwiftDemucs`) and
exposes it to [`mlx-engine-swift`](https://github.com/xocialize/mlx-engine-swift) as a
`ModelPackage`. All model logic — hybrid spectral/temporal branches, cross-domain transformer,
chunked overlap-add — lives in the core; this package maps the canonical
`AudioSeparationRequest → [Stem: Audio]` contract onto it and owns the engine lifecycle.

## Capability

| | |
|---|---|
| Capability | `audioSeparation` |
| Input | mixture `Audio` (.wav, any rate/channels — resampled to 44.1 kHz stereo) |
| Output | `[Stem: Audio]` — `.vocals` and `.instrumental` (.wav, 44.1 kHz stereo) |

The `htdemucs_ft` vocal branch estimates the vocal stem; the instrumental is its complement.

## Weights

`mlx-community/htdemucs-ft-vocals-mlx` (`htdemucs_ft_vocals.safetensors`, fp16), selected via
`DemucsConfiguration.repo` (default).

## Usage

```swift
import MLXServeCore
import MLXDemucs

let engine = MLXServeEngine()
try await engine.register(DemucsSeparationPackage.registration, configuration: DemucsConfiguration())

let mixture = Audio(format: .wav, data: wavBytes)
let response = try await engine.run(AudioSeparationRequest(audio: mixture)) as! AudioSeparationResponse
let vocals = response[.vocals]
let instrumental = response[.instrumental]
```

## Consuming it

Public + version-tagged on github.com/xocialize. Add by tagged URL:
`.package(url: "https://github.com/xocialize/mlx-demucs-swift", from: "0.1.0")`, then import `MLXDemucs` (the conformant `audioSeparation` package). Builds standalone — its engine contract (`MLXToolKit`) and model-core dependencies are tagged-URL net deps, no local checkouts.

Requirements: macOS 26+ (Apple Silicon, Metal GPU). Port code MIT; weights MIT (Meta Demucs).
