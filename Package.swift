// swift-tools-version: 6.2
import PackageDescription

// mlx-demucs-swift — the MLXEngine `audioSeparation` package over HTDemucs v4.
// A thin conformance layer over the standalone inference engine demucs-mlx-swift (product
// `SwiftDemucs`), the same pattern as mlx-mel-roformer-swift / mlx-voxcpm2-tts-swift. The engine
// contract (MLXToolKit) is a local-path dep for in-workspace dev; the SwiftDemucs core is pinned
// to a tagged release. Swift-port naming: `-swift` on the package/repo; module/product is `MLXDemucs`.
let package = Package(
    name: "mlx-demucs-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXDemucs", targets: ["MLXDemucs"]),
    ],
    dependencies: [
        .package(path: "../mlx-engine-swift"),
        .package(url: "https://github.com/xocialize/demucs-mlx-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "MLXDemucs",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                // Package identity derives from the repo URL's last path component.
                .product(name: "SwiftDemucs", package: "demucs-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            // SwiftDemucs' `VocalSeparator` (MLX) isn't Sendable-audited, so awaiting its nonisolated
            // init back into the `@InferenceActor` trips strict region-isolation. The engine serializes
            // all lifecycle on InferenceActor (no real concurrency), so v5 mode keeps that a warning —
            // same lever as Kokoro / VoxCPM2 / Mel-RoFormer.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXDemucsTests",
            dependencies: [
                "MLXDemucs",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
