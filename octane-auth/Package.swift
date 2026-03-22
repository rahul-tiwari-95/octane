// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "octane-auth",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "octane-auth",
            path: "Sources",
            linkerSettings: [
                .linkedFramework("LocalAuthentication"),
                .linkedFramework("Security"),
                .linkedFramework("Foundation"),
            ]
        )
    ]
)
