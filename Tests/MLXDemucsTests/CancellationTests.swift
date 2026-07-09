// CancellationTests.swift — HTDemucs separation through the engine's CAN gate (offline, no MLX
// kernels). CAN-1/2 drive the real run() pre-cancelled: the entry checkpoint fires before the
// notLoaded guard or weights, so a stub configuration suffices. CAN-3 is the document of record
// for the checkpoint cadence: the SwiftDemucs core's overlap-add loop checks cancellation once
// per 30 s chunk (`checkCancelled()` in `VocalSeparator.separateChunked`, demucs-mlx-swift
// ≥ 0.1.1 — task lane throws CancellationError unchanged) and the wrapper adds no catch that
// could launder it.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXDemucs

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = DemucsSeparationPackage(configuration: DemucsConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: AudioSeparationRequest(audio: Audio(format: .wav, data: Data(),
                                                         sampleRate: 44_100, channels: 2)))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // audioSeparation is not a long-run capability by name, but the declared 2.2 GB peak
        // activation (≥ 2 GB) implies long runs — the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: DemucsSeparationPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: DemucsSeparationPackage.manifest,
            posture: .cadence([
                // The core processes the mixture in fixed 30 s chunks (0.5 s overlap-add) and
                // checks cancellation once per chunk: `try checkCancelled()` at the top of the
                // while-loop in `VocalSeparator.separateChunked` (SwiftDemucs core). Each chunk
                // is one full spectral+temporal+transformer forward producing the separated
                // stem — the run's generative phase — so `generate/chunk` is the honest single
                // entry (there is no separately checkpointed encode or decode stage).
                // `separate(samples:)` takes no progress handler, so no RunProgress is reported.
                .init(phase: .generate, unit: .chunk),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
