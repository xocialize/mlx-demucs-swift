import Testing
import Foundation
import MLXToolKit
@testable import MLXDemucs

/// Offline conformance checks — no weights, no Metal. Live separation + weight-parity is proven
/// in the `MLXEngine Testing` app (the model needs the GPU and the published checkpoint).
struct DemucsSeparationTests {

    @Test func manifestIsAudioSeparationAndPermissive() {
        let m = DemucsSeparationPackage.manifest
        #expect(m.capabilities == [.audioSeparation])
        #expect(m.license.weightLicense == .mit)
        #expect(m.license.portCodeLicense == .mit)
        #expect(m.provenance.sourceRepo == "mlx-community/htdemucs-ft-vocals-mlx")
    }

    @Test func manifestRequirements() {
        let r = DemucsSeparationPackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
        #expect(r.footprints.first?.quant == .fp16)
    }

    @Test func surfaceIsTheCanonicalSeparationDescriptor() {
        let surface = DemucsSeparationPackage.manifest.surfaces.first
        #expect(surface?.capability == .audioSeparation)
        #expect(surface?.parameters.first?.kind == .audio)
    }

    @Test func registrationConstructs() throws {
        let reg = DemucsSeparationPackage.registration
        #expect(reg.manifest.capabilities == [.audioSeparation])
        let pkg = try reg.makePackage(DemucsConfiguration())
        #expect(pkg is DemucsSeparationPackage)
    }

    @Test func configurationDefaultsToPublishedRepo() {
        #expect(DemucsConfiguration().repo == "mlx-community/htdemucs-ft-vocals-mlx")
    }

    @Test func configurationCodableExcludesEnvironmentRoot() throws {
        var c = DemucsConfiguration()
        c.modelsRootDirectory = URL(fileURLWithPath: "/tmp/should-not-persist")
        let data = try JSONEncoder().encode(c)
        let back = try JSONDecoder().decode(DemucsConfiguration.self, from: data)
        #expect(back.repo == "mlx-community/htdemucs-ft-vocals-mlx")
        #expect(back.modelsRootDirectory == nil)
    }

    @Test func wavHeaderIsValidStereo() {
        let interleaved = [Float](repeating: 0, count: 200)  // 100 stereo frames
        let wav = DemucsSeparationPackage.encodeWAV16(interleaved: interleaved, channels: 2, sampleRate: 44_100)
        #expect(wav.count == 44 + 400)
        #expect(wav.prefix(4) == Data("RIFF".utf8))
        #expect(wav[8..<12] == Data("WAVE".utf8))
        #expect(wav[36..<40] == Data("data".utf8))
        #expect(wav[22] == 2)  // num channels
    }
}
