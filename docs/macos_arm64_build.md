# Building stable-retro on Apple Silicon (macOS, arm64)

This is the part no tutorial covers: getting `stable-retro` to actually compile
and run on a recent Apple Silicon Mac. We hit three separate, non-obvious
blockers. They are all solved by [`scripts/setup_stable_retro.sh`](../scripts/setup_stable_retro.sh);
this document explains *why* each step is there.

Verified on: macOS 26.3 (arm64), Apple clang 21, Python 3.12, stable-retro `main`.

---

## TL;DR

```bash
brew install cmake pkg-config lua@5.4 libzip capnp
git clone --depth 1 https://github.com/Farama-Foundation/stable-retro.git
cd stable-retro
# comment out every add_core(...) in CMakeLists.txt EXCEPT add_core(nes fceumm)
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
pip install -e .
```

Or just run our script, which does exactly this idempotently.

---

## Blocker 1 — the published arm64 wheel is mislabeled

`pip install stable-retro` *appears* to work, then explodes at import:

```
ImportError: dlopen(.../_retro.cpython-312-darwin.so):
  tried: '...' (mach-o file, but is an incompatible architecture
  (have 'x86_64', need 'arm64'))
```

The PyPI wheel `stable_retro-1.0.0-cp312-cp312-macosx_11_0_arm64.whl` is tagged
`arm64` but **contains an x86_64 binary**. We confirmed this by unzipping the
wheel and running `file` on the `.so` — it reports `x86_64`. There is no working
prebuilt arm64 binary on PyPI, conda-forge, or anaconda.org.

**Consequence:** on Apple Silicon you must build from source.

## Blocker 2 — `lua@5.3` no longer exists in Homebrew

The official macOS guide says `brew install ... lua@5.3`. Homebrew has since
dropped that formula; only `lua@5.4` is available. This is harmless: Life Force
uses a declarative `scenario.json` (no Lua scripting), so Lua is only a
build-time dependency. **Use `lua@5.4`.**

## Blocker 3 — modern Apple clang can't compile the old vendored cores

With the build dependencies present, cmake configures fine, but the compile
fails inside the **bundled zlib** of the **Genesis** and **PCE** cores:

```
.../MacOSX.sdk/usr/include/_stdio.h:322:7: error: expected identifier or '('
const char * ZEXPORT zError(err)        <- K&R-style definition
make[3]: *** [.../genesis_plus_gx_libretro.dylib] Error 2
make[3]: *** [.../mednafen_pce_fast_libretro.dylib] Error 2
```

Those cores vendor a very old zlib written in pre-C23 K&R C, which Apple clang 21
(defaulting to a newer C standard) rejects. The maintainers anticipated this and
set `-std=gnu17` for the cores in `CMakeLists.txt`, but that flag does not reach
the *nested* zlib sub-builds, so Genesis/PCE still fail.

**The key realization:** those are the Genesis and PCE cores. **Life Force is an
NES game — we only need the NES core (`fceumm`), which compiles cleanly as native
arm64.** The Python extension links `retro-base`, and `retro-base` has a
build-ordering dependency on *every* core (`add_dependencies(retro-base
${CORE_TARGETS})`), so one broken core fails the whole build — even though cores
are loaded at *runtime* and the extension doesn't truly need them at build time.

**Fix:** register only the NES core. In `CMakeLists.txt`, comment out every
`add_core(...)` line except `add_core(nes fceumm)`. Then `retro-base` depends
only on `nes`, and the build succeeds.

---

## Verification

After a successful build:

```python
import stable_retro as retro
print(retro.data.list_games().__contains__("LifeForce-Nes-v0"))  # True
# fceumm_libretro.dylib should be present and report arm64 via `file`
```

A `retro.make("LifeForce-Nes-v0")` before importing a ROM raises
`FileNotFoundError: Game not found ... Did you make sure to import the ROM?` —
that is the expected "everything works, just bring the ROM" state.

## Known minor issue — pyglet window teardown on macOS

Calling `env.close()` after a window has been opened can raise:

```
AttributeError: 'CocoaAlternateEventLoop' object has no attribute 'platform_event_loop'
```

This is a pyglet/Cocoa teardown bug that fires *after* all real work completes.
It does not affect training (which renders to `rgb_array` / records video and
runs envs in subprocesses). Avoid the live `human` viewer on macOS; record video
instead.
