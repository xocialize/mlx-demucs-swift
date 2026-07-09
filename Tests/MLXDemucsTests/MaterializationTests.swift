// MaterializationTests.swift — Demucs through the engine's MAT gate (offline, no network):
// the WeightSourcing declaration, fresh-machine honesty, explicit-path satisfaction, and the
// store-layout probe/resolution. Single fp16 checkpoint — one declaration covers the package.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXDemucs

final class MaterializationTests: XCTestCase {

    /// Temp dir holding a probe file that makes an explicit-dir config read as satisfied.
    private func satisfiedDir() throws -> (dir: URL, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "demucs-mat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: dir.appending(path: DemucsConfiguration.weightsFile).path, contents: Data([0]))
        return (dir, { try? FileManager.default.removeItem(at: dir) })
    }

    // MARK: - Engine MAT gate

    func testMATGate() throws {
        let (dir, cleanup) = try satisfiedDir()
        defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: DemucsConfiguration(),
            satisfiedConfiguration: DemucsConfiguration(modelDirectory: dir))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Source declaration shape

    func testDeclaresSingleMainSource() {
        let sources = DemucsConfiguration().weightSources
        XCTAssertEqual(sources.map(\.role), ["main"])
        XCTAssertEqual(sources[0].repo, "mlx-community/htdemucs-ft-vocals-mlx")
        XCTAssertEqual(sources[0].matching, ["htdemucs_ft_vocals.safetensors"])
    }

    // MARK: - Store-layout probe + resolution

    func testStoreLayoutSatisfiesAndResolves() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "demucs-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = DemucsConfiguration()
        // Empty store: the source is missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        // Populate the expected layout.
        let dir = root.appending(path: cfg.repo)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: dir.appending(path: DemucsConfiguration.weightsFile).path, contents: Data([0]))
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        // Resolution lands on the store layout; an explicit dir always wins.
        XCTAssertEqual(cfg.resolved(storeRoot: root).modelDirectory?.path, dir.path)
        let explicit = DemucsConfiguration(modelDirectory: URL(fileURLWithPath: "/x"))
            .resolved(storeRoot: root)
        XCTAssertEqual(explicit.modelDirectory?.path, "/x")
    }

    func testPrewarmPathsUseResolvedStoreLayout() {
        let root = URL(fileURLWithPath: "/tmp/some-store")
        let cfg = DemucsConfiguration(modelsRootDirectory: root)
        XCTAssertEqual(
            cfg.prewarmPaths.map(\.path),
            [root.appending(
                path: "mlx-community/htdemucs-ft-vocals-mlx/htdemucs_ft_vocals.safetensors").path])
    }

    func testCodableRoundTrip() throws {
        let cfg = DemucsConfiguration(modelDirectory: URL(fileURLWithPath: "/x"))
        let decoded = try JSONDecoder().decode(DemucsConfiguration.self,
                                               from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.repo, cfg.repo)
        XCTAssertNil(decoded.modelDirectory)   // environment-specific, never encoded
    }
}
