# Tile break probe

`tile_break_probe.cpp` measures what ending and resuming a render pass costs on a GPU. A tile-based
GPU (Adreno, Mali) stores every attachment when a pass ends and loads it back when the pass
resumes; a desktop GPU does neither. Mesh emulation ends the pass before each task or mesh draw to
run the program as compute dispatches, so on a phone its cost is the break, not the dispatch.

The probe draws the same frame three ways into an offscreen target (colour attachments and depth,
load and store, general layout, as the emulator's passes) and times each submission on the GPU:

- the dispatches before one pass, which is what `EDEN_NX2_MESH_PREPASS=1` records,
- a pass break before each draw, which is what mesh emulation records without it,
- one pass without dispatches.

It prints the three times, the cost of one break, and whether the three images match.

Android, with the NDK (API 31 or later):

```sh
$NDK/toolchains/llvm/prebuilt/<host>/bin/aarch64-linux-android31-clang++ -std=c++20 -O2 \
    -static-libstdc++ tools/nx2/tile_break_probe.cpp -o tile_break_probe
adb shell mkdir -p /data/local/tmp/probe
adb push tile_break_probe /data/local/tmp/probe/
adb shell "cd /data/local/tmp/probe && chmod 755 tile_break_probe && ./tile_break_probe"
```

Linux: `clang++ -std=c++20 -O2 tools/nx2/tile_break_probe.cpp -o tile_break_probe`.

Options: `--width`, `--height` (1920x1080), `--colors` (3, up to 4), `--half-float` (RGBA16F
instead of RGBA8), `--draws` (64 a frame), `--frames` (20 measured, after 3 warm-up frames).

`--icd <driver .so>` uses that driver through its `vk_icdGetInstanceProcAddr` instead of the system
loader, to measure a driver the system doesn't know: push a Turnip build's `libvulkan_freedreno.so`
(from its adrenotools zip) next to the probe and pass its path. Mesa reads its switches from the
environment, so the same driver compares its tiled and direct paths:

```sh
adb shell "cd /data/local/tmp/probe && TU_DEBUG=gmem ./tile_break_probe --icd ./libvulkan_freedreno.so"
adb shell "cd /data/local/tmp/probe && TU_DEBUG=sysmem ./tile_break_probe --icd ./libvulkan_freedreno.so"
```

## Reading the result

Multiply the break cost by the breaks a frame the game makes. `EDEN_NX2_GPU_PROFILE=1` logs them
per second and call site (`Render pass ends <n>/s at vk_graphics_pipeline.cpp:<line>`, the mesh
emulation site is the `RequestOutsideRenderPassOperationContext` in `ConfigureDraw`); divide by the
frames per second. Use the game's target size and attachment count: the cost grows with both.
