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
    /// Explicit weights directory (dev escape hatch — never touches the network).
    public var modelDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Set by the engine from its
    /// `ModelStore`. Excluded from `Codable` (environment-specific).
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/htdemucs-ft-vocals-mlx",
                modelDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension DemucsConfiguration: WeightSourcing {
    /// The single checkpoint `VocalSeparator(weightsDirectory:)` reads.
    static let weightsFile = "htdemucs_ft_vocals.safetensors"

    public var weightSources: [WeightSource] {
        [WeightSource(role: "main", repo: repo, matching: [Self.weightsFile])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        // Explicit local directory first (dev escape hatch).
        if let dir = modelDirectory,
           fm.fileExists(atPath: dir.appending(path: Self.weightsFile).path) {
            return []
        }
        // Then the ModelStore layout (`<root>/<org>/<name>`).
        if let dir = ModelStore(root: storeRoot).directory(for: repo),
           fm.fileExists(atPath: dir.appending(path: Self.weightsFile).path) {
            return []
        }
        return weightSources
    }

    /// The configuration with a nil `modelDirectory` resolved to the store layout — what `load()`
    /// uses AFTER materialization. An explicit directory always wins.
    public func resolved(storeRoot: URL?) -> DemucsConfiguration {
        var cfg = self
        if cfg.modelDirectory == nil {
            cfg.modelDirectory = ModelStore(root: storeRoot).directory(for: repo)
        }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension DemucsConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        // Store-resolved checkpoint path; the prewarmer skips it when absent (first launch).
        guard let dir = resolved(storeRoot: modelsRootDirectory).modelDirectory else { return [] }
        return [dir.appending(path: Self.weightsFile)]
    }
}
