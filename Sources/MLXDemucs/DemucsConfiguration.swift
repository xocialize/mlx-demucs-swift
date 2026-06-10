import Foundation
import MLXToolKit

/// Init-time configuration for `DemucsSeparationPackage` (C9): which published checkpoint to load.
/// Per-request input/stems ride the `AudioSeparationRequest`, not here.
public struct DemucsConfiguration: PackageConfiguration, ModelStorable {
    /// HuggingFace repo holding `htdemucs_ft_vocals.safetensors`.
    public var repo: String
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
