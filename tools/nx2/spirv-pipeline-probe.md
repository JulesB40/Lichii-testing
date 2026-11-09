# SPIR-V pipeline probe

`spirv_pipeline_probe.cpp` builds a compute pipeline from each SPIR-V module given and prints the
driver's result. Use it when a driver refuses a shader (`Mesh emulation mesh-compute pipeline build
failed`, `Graphics pipeline build failed: VK_ERROR_UNKNOWN`) and the log says nothing more: take the
module from `EDEN_NX2_SHADER_DUMP` (`<shader hash>-<stage>.spv`), confirm the probe fails on it, then
edit its disassembly (`spirv-dis`, `spirv-as --target-env vulkan1.3`) until it builds. The last edit
names the construct the driver can't take.

The descriptor set layouts come from the module's bindings, push constants get the device's whole
range, and every Vulkan 1.1-1.3 core feature the device supports is enabled. Only compute modules
(compute, and mesh and task programs as mesh emulation builds them) are supported.

Android, with the NDK (API 31 or later):

```sh
$NDK/toolchains/llvm/prebuilt/<host>/bin/aarch64-linux-android31-clang++ -std=c++20 -O2 \
    -static-libstdc++ tools/nx2/spirv_pipeline_probe.cpp -lvulkan -o spirv_pipeline_probe
adb shell mkdir -p /data/local/tmp/probe
adb push spirv_pipeline_probe dump /data/local/tmp/probe/
adb shell "cd /data/local/tmp/probe && chmod 755 spirv_pipeline_probe && ./spirv_pipeline_probe dump/*.spv"
```

Each module prints one line, `<path>: VK_SUCCESS (0), ...` or the error, then `<n> of <m> failed`.
