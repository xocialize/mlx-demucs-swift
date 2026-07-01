# Efficiency Adoption Brief — `mlx-demucs-swift` (HTDemucs v4, `audioSeparation`)

> **For a session-specific agent.** Adopt engine 1.14 efficiency (engine 0.17.0+). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (four levers + **"Measurement
> findings"**) + references/memory-harness.md. LIGHT **split + unload-clearCache** adoption. Audited 2026-06-30.

## Package at a glance
- Wrapper `MLXDemucs` (`DemucsSeparationPackage: ModelPackage`). Capability **`audioSeparation`** (HTDemucs v4
  stem separation). Engine pinned `from: "0.3.0"`. Single component: `separator: VocalSeparator?`.
- **Footprint today (FLAT residentBytes only, NO transient):** `QuantFootprint(.fp16, 2.5 GB)`.
- `unload()` (~line 64) — verify it `MLX.Memory.clearCache()`s (the grep says no).

## Audit vs. the four levers
| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.3.0 → 0.17.0 | **P0** |
| 1. Split footprint | ❌ | flat 2.5 GB, no transient | **P1 (headline)** |
| 2. Per-stage evict | ➖ N/A | single separator model | note |
| 3. mmap/lazy | 🟡 verify | confirm lazy load (floor ≈ on-disk) | note |
| 4. BudgetAware | ➖ | single quant | defer |

## Plan
- **P0:** `swift package update` → 0.17.0; build + fix any drift.
- **P1 (headline):** split the flat 2.5 GB. `residentBytes` = the separator weights floor; `peakActivationBytes`
  = the per-**segment** transient. HTDemucs processes audio in fixed **segments/chunks** → the activation is
  segment-bounded (like a tile, not whole-clip-driven) — measure at the default segment size and note it as the
  activation driver (the Real-ESRGAN tile lesson). Adopt `QuantConfigured` (single fp16).
- **`unload()` must `MLX.Memory.clearCache()`** after niling `separator`.

## Measurement
Declare `residentBytes` from the measured weight floor (solid) + a **FLAGGED** `peakActivationBytes` from the
package smoke (in-app phys reads ~2.5–2.9× higher — admission basis). Audio is lighter than video — a smoke at
the default segment is tractable; still flag pending an in-app phys re-baseline.

## Definition of done
- [ ] engine 0.17.0; `QuantConfigured`; P1 split (note the segment size as the activation driver); `unload()` clearCache.
- [ ] Smoke green (valid stem separation); split recorded; activation flagged.
- [ ] Registry: demucs row Eff ⬜→✅ (note "activation = smoke est, phys re-baseline pending"), Eng→0.17.0.

## Report back
flat→split, the segment-bounded transient, drift since 0.3.0, effort, commit SHA. STAY IN SCOPE — four-lever
adoption + this brief + registry row only; verify `git show --stat`; stop-and-report if bigger.
