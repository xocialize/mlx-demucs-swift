import Foundation
import MLXToolKit

/// Init-time configuration for `DemucsSeparationPackage` (C9): which published checkpoint to load.
/// Per-request input/stems ride the `AudioSeparationRequest`, not here.
///
/// Conforms to `QuantConfigured` (single fp16 checkpoint) so the memory governor charges the
/// declared fp16 `QuantFootprint` (split resident + activation) directly instead of the
/// largest-that-fits heuristic.
public struct DemucsConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// HuggingFace repo holding `htdemucs_ft_vocals.safetensors`.
    public var repo: String

    /// Shipped quantization of the published checkpoint (fp16). Exposed for `QuantConfigured` so
    /// the governor matches the fp16 footprint.
    public var quant: Quant { .fp16 }
    /// Where weights are materialized. Set by the engine from its `ModelStore`; `nil` → the
    /// default swift-transformers cache. Excluded from `Codable` (environment-specific).
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/htdemucs-ft-vocals-mlx",
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo
    }
}
