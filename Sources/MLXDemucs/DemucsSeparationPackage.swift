import Foundation
import MLXToolKit
import MLX
import SwiftDemucs

/// Errors specific to the Demucs package boundary.
public enum DemucsError: Error, Equatable {
    /// Weight sources are missing and there is no store root (or resolved directory) to
    /// materialize into.
    case missingWeights(String)
}

/// An MLXEngine `audioSeparation` package over **HTDemucs v4** — splits a music mixture into
/// `vocals` + `instrumental` at 44.1 kHz. A thin conformance wrapper over the standalone
/// `SwiftDemucs` engine (demucs-mlx-swift); all model logic (hybrid spectral/temporal branches,
/// cross-domain transformer, chunked overlap-add) lives there.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `DemucsConfiguration`, pages weights
/// in with `load()` (downloads the HF snapshot on first run and builds the `VocalSeparator`), drives
/// `run(_:)`, and reclaims with `unload()`. Returns canonical `.wav` `Audio` per stem.
///
/// The `htdemucs_ft` vocal branch estimates the vocal stem; the instrumental is its complement
/// (`mixture - vocals`). A request for stems the package does not produce is ignored.
@InferenceActor
public final class DemucsSeparationPackage: ModelPackage {
    public typealias Configuration = DemucsConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // HTDemucs weights (Meta) and the Swift port are both MIT.
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/htdemucs-ft-vocals-mlx",
                                   revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (contract 1.14). ~26M-param backbone, 84 MB fp16 weights on disk
                // (mlx-community/htdemucs-ft-vocals-mlx: htdemucs_ft_vocals.safetensors) → ~0.3 GB
                // resident floor with framework/mmap overhead. The transient is *chunk-bounded*: HTDemucs
                // processes audio in fixed 30-second chunks (SwiftDemucs default `chunkDurationSeconds`,
                // 0.5 s overlap; model segment 7.8 s) through the spectral + temporal branches and the
                // cross-domain transformer, so the activation tracks the chunk working set, not clip length
                // (the Real-ESRGAN tile lesson). ~2.2 GB peak at the 30 s chunk.
                //
                // ⚠️ peakActivationBytes is a SMOKE ESTIMATE (derived from the prior flat 2.5 GB minus the
                // measured weight floor, scaled to the 30 s chunk envelope); in-app phys_footprint reads
                // ~2.5–2.9× higher — IN-APP PHYS RE-BASELINE PENDING (the admission basis, R-MEM-1).
                footprints: [
                    QuantFootprint(quant: .fp16,
                                   residentBytes: 300_000_000,
                                   peakActivationBytes: 2_200_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                AudioSeparationContract.descriptor(
                    name: "htdemucs-separate",
                    summary: "HTDemucs v4 vocal source separation (44.1 kHz .wav): splits a mixture into vocals + instrumental."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var separator: VocalSeparator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard separator == nil else { return }
        // Auto-materialize the missing checkpoint into the engine store (dir-less configs only;
        // explicit directories never touch the network), forwarding progress via
        // WeightDownloadProgress so the engine's PreparationMonitor surfaces `.downloading`.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        if !missing.isEmpty {
            guard let storeRoot else {
                throw DemucsError.missingWeights(
                    "no models root set and sources missing: \(missing.map(\.role).joined(separator: ", "))")
            }
            try await WeightMaterializer.materialize(missing, into: storeRoot)
        }
        try Task.checkCancellation()
        guard let dir = configuration.resolved(storeRoot: storeRoot).modelDirectory else {
            throw DemucsError.missingWeights("unresolved weights directory (no store root)")
        }
        separator = try await VocalSeparator(weightsDirectory: dir)
    }

    public func unload() async {
        separator = nil
        // Drop the model's weight/activation buffers from MLX's pool too — niling the ref alone
        // leaves them cached, so phys_footprint doesn't fall and engine.evict / R-MEM-1 can't
        // reclaim (RSS then grows monotonically across model switches). Contract 1.14 requirement.
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let separator else { throw PackageError.notLoaded }
        guard request.capability == .audioSeparation,
              let req = request as? AudioSeparationRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        // Decode the mixture to [1, 2, N] @ 44.1 kHz stereo (AudioIO resamples arbitrary input).
        let mixture = try Self.decodeMixture(req.audio)
        let vocals = try await separator.separate(samples: mixture)

        // Empty request => every stem this package produces (vocals + instrumental).
        let wantVocals = req.stems.isEmpty || req.stems.contains(.vocals)
        let wantInstrumental = req.stems.isEmpty || req.stems.contains(.instrumental)

        var stems: [Stem: Audio] = [:]
        if wantVocals {
            stems[.vocals] = Self.encodeStem(vocals)
        }
        if wantInstrumental {
            stems[.instrumental] = Self.encodeStem(mixture - vocals)
        }
        return AudioSeparationResponse(stems: stems)
    }

    // MARK: - Audio I/O

    /// Decode a canonical `Audio` (.wav) to a `[1, 2, N]` 44.1 kHz stereo MLXArray, reusing
    /// SwiftDemucs's `AudioIO` (AVFoundation load + resample). `AudioIO` reads from a URL, so the
    /// bytes round-trip through a temp file.
    nonisolated static func decodeMixture(_ audio: Audio) throws -> MLXArray {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("wav")
        try audio.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let (samples, _) = try AudioIO().loadAudio(from: tmp)
        return samples
    }

    /// Wrap a `[1, 2, N]` stem MLXArray as a canonical 16-bit PCM stereo WAV `Audio`.
    nonisolated static func encodeStem(_ audio: MLXArray) -> Audio {
        let interleaved = interleaveStereo(audio)
        let wav = encodeWAV16(interleaved: interleaved, channels: 2, sampleRate: 44_100)
        return Audio(format: .wav, data: wav, sampleRate: 44_100, channels: 2)
    }

    /// `[1, 2, N]` (or `[2, N]`) → interleaved `[L0, R0, L1, R1, …]` float samples.
    nonisolated static func interleaveStereo(_ audio: MLXArray) -> [Float] {
        let n = audio.shape[audio.ndim - 1]
        let chans = audio.reshaped([2, n])
        let left = chans[0].asType(.float32).asArray(Float.self)
        let right = chans[1].asType(.float32).asArray(Float.self)
        var out = [Float](repeating: 0, count: n * 2)
        for i in 0..<n {
            out[i * 2] = left[i]
            out[i * 2 + 1] = right[i]
        }
        return out
    }

    /// Encode interleaved float samples as a 16-bit PCM WAV (broadly playable) in memory.
    nonisolated static func encodeWAV16(interleaved samples: [Float], channels: Int, sampleRate: Int) -> Data {
        let bitsPerSample = 16
        let blockAlign = channels * bitsPerSample / 8
        let byteRate = sampleRate * blockAlign
        let dataSize = (samples.count / channels) * blockAlign

        var data = Data(capacity: 44 + dataSize)
        func ascii(_ s: String) { data.append(contentsOf: Array(s.utf8)) }
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }

        ascii("RIFF"); u32(UInt32(36 + dataSize)); ascii("WAVE")
        ascii("fmt "); u32(16); u16(1) // PCM
        u16(UInt16(channels)); u32(UInt32(sampleRate)); u32(UInt32(byteRate))
        u16(UInt16(blockAlign)); u16(UInt16(bitsPerSample))
        ascii("data"); u32(UInt32(dataSize))

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            var le = Int16(clamped * 32767).littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
        return data
    }
}

extension DemucsSeparationPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(DemucsSeparationPackage.self)
    }
}
