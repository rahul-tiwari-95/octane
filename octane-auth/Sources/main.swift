import Foundation
import LocalAuthentication
import Security

// ── Helpers ───────────────────────────────────────────────────────────────────

let SERVICE = "com.srswti.octane"

func jsonOK(_ extras: [String: Any] = [:]) {
    var d: [String: Any] = ["success": true]
    extras.forEach { d[$0] = $1 }
    printJSON(d)
}

func jsonError(_ msg: String) {
    printJSON(["success": false, "error": msg])
}

func printJSON(_ d: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: d),
       let s = String(data: data, encoding: .utf8) {
        print(s)
    }
    exit(d["success"] as? Bool == true ? 0 : 1)
}

func account(_ vault: String, _ key: String) -> String { "\(vault).\(key)" }

// ── Commands ──────────────────────────────────────────────────────────────────

/// Store a secret value in Keychain protected by Touch ID (biometryCurrentSet).
/// Re-enrolling fingerprints invalidates any existing items — this is intentional.
func storeSecret(vault: String, key: String, value: String) {
    var cfErr: Unmanaged<CFError>?
    guard let access = SecAccessControlCreateWithFlags(
        kCFAllocatorDefault,
        kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly as CFTypeRef,
        .biometryCurrentSet,
        &cfErr
    ) else {
        let msg = cfErr.map { "\($0.takeRetainedValue())" } ?? "Unknown"
        jsonError("SecAccessControlCreateWithFlags failed: \(msg)")
        return
    }

    guard let valueData = value.data(using: .utf8) else {
        jsonError("Failed to encode value as UTF-8")
        return
    }

    let acct = account(vault, key)

    // Delete existing item first (ignore errors — it may not exist)
    let deleteQuery: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: SERVICE,
        kSecAttrAccount as String: acct,
    ]
    SecItemDelete(deleteQuery as CFDictionary)

    // Add new item with biometry protection
    let addQuery: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: SERVICE,
        kSecAttrAccount as String: acct,
        kSecValueData as String: valueData,
        kSecAttrAccessControl as String: access,
        kSecAttrLabel as String: "Octane vault: \(vault)",
    ]

    let status = SecItemAdd(addQuery as CFDictionary, nil)
    if status == errSecSuccess {
        jsonOK()
    } else {
        jsonError("SecItemAdd failed: OSStatus \(status)")
    }
}

/// Retrieve a secret value from Keychain, triggering Touch ID authentication.
func retrieveSecret(vault: String, key: String, reason: String) {
    let context = LAContext()
    context.localizedReason = reason

    let query: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: SERVICE,
        kSecAttrAccount as String: account(vault, key),
        kSecReturnData as String: true,
        kSecMatchLimit as String: kSecMatchLimitOne,
        kSecUseAuthenticationContext as String: context,
    ]

    var result: AnyObject?
    let status = SecItemCopyMatching(query as CFDictionary, &result)

    if status == errSecSuccess,
       let data = result as? Data,
       let value = String(data: data, encoding: .utf8) {
        jsonOK(["value": value])
    } else if status == errSecItemNotFound {
        jsonError("Item not found in Keychain. Run 'octane vault create \(vault)' first.")
    } else if status == errSecUserCanceled || status == errSecAuthFailed {
        jsonError("Touch ID authentication cancelled or failed.")
    } else {
        jsonError("SecItemCopyMatching failed: OSStatus \(status)")
    }
}

/// Check that Touch ID is available and verify identity (no data returned).
func checkBiometry(vault: String) {
    let ctx = LAContext()
    var laErr: NSError?
    guard ctx.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &laErr) else {
        jsonError("Touch ID not available: \(laErr?.localizedDescription ?? "Unknown")")
        return
    }

    ctx.evaluatePolicy(
        .deviceOwnerAuthenticationWithBiometrics,
        localizedReason: "Octane: Verify access to \(vault) vault"
    ) { success, err in
        if success {
            jsonOK()
        } else {
            jsonError(err?.localizedDescription ?? "Authentication failed")
        }
    }

    // Keep run loop alive until async completion
    RunLoop.main.run(until: Date(timeIntervalSinceNow: 30))
}

/// Delete a Keychain item permanently.
func deleteSecret(vault: String, key: String) {
    let deleteQuery: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: SERVICE,
        kSecAttrAccount as String: account(vault, key),
    ]
    let status = SecItemDelete(deleteQuery as CFDictionary)
    if status == errSecSuccess || status == errSecItemNotFound {
        jsonOK()
    } else {
        jsonError("SecItemDelete failed: OSStatus \(status)")
    }
}

/// List all Keychain items created by Octane (by service label).
func listVaults() {
    let query: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: SERVICE,
        kSecReturnAttributes as String: true,
        kSecMatchLimit as String: kSecMatchLimitAll,
    ]

    var result: AnyObject?
    let status = SecItemCopyMatching(query as CFDictionary, &result)

    if status == errSecItemNotFound {
        jsonOK(["vaults": [String](), "count": 0])
        return
    }

    guard status == errSecSuccess, let items = result as? [[String: Any]] else {
        jsonError("SecItemCopyMatching failed: OSStatus \(status)")
        return
    }

    // Extract vault names from account field (format: vault.key)
    var vaultSet = Set<String>()
    for item in items {
        if let acct = item[kSecAttrAccount as String] as? String {
            let parts = acct.split(separator: ".", maxSplits: 1)
            if let vaultName = parts.first {
                vaultSet.insert(String(vaultName))
            }
        }
    }
    jsonOK(["vaults": Array(vaultSet).sorted(), "count": vaultSet.count])
}

// ── Entry point ───────────────────────────────────────────────────────────────

let args = CommandLine.arguments

if args.count < 2 {
    fputs("""
    Usage:
      octane-auth store    <vault> <key> <value>
      octane-auth retrieve <vault> <key> [reason]
      octane-auth check    <vault>
      octane-auth delete   <vault> <key>
      octane-auth list-vaults
    
    Output: JSON {success: bool, value?: str, error?: str}
    Exit code: 0 on success, 1 on failure.
    """, stderr)
    exit(2)
}

let cmd = args[1]

switch cmd {
case "store":
    guard args.count >= 5 else { jsonError("store requires: <vault> <key> <value>"); break }
    storeSecret(vault: args[2], key: args[3], value: args[4])

case "retrieve":
    guard args.count >= 4 else { jsonError("retrieve requires: <vault> <key>"); break }
    let reason = args.count >= 5 ? args[4] : "Octane needs access to the \(args[2]) vault."
    retrieveSecret(vault: args[2], key: args[3], reason: reason)

case "check":
    guard args.count >= 3 else { jsonError("check requires: <vault>"); break }
    checkBiometry(vault: args[2])

case "delete":
    guard args.count >= 4 else { jsonError("delete requires: <vault> <key>"); break }
    deleteSecret(vault: args[2], key: args[3])

case "list-vaults":
    listVaults()

default:
    jsonError("Unknown command: \(cmd). Use store|retrieve|check|delete|list-vaults.")
}
