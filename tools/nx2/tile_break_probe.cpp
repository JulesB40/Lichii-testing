// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Measures what ending and resuming a render pass costs on this GPU. A tile-based GPU (Adreno,
// Mali) stores every attachment when a pass ends and loads it back when the pass resumes, and
// mesh emulation ends the pass before each task or mesh draw to run the program as a compute
// dispatch. The probe draws the same frame three ways into an offscreen target and times the GPU:
// the dispatches before one pass, a pass break before each draw, and one pass without dispatches.
//
// Android: build with the NDK's clang (see tile-break-probe.md), push it to /data/local/tmp and
// run it from adb shell. --icd <driver .so> measures a driver the system loader doesn't know
// (a Turnip build), through its vk_icdGetInstanceProcAddr.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <dlfcn.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

namespace {

// probe.comp: writes where draw `index` of `count` goes (a cell of a grid) to centers[index].
//   layout(local_size_x = 1) in;
//   layout(push_constant) uniform Constants { uint index; uint count; } pc;
//   layout(set = 0, binding = 0) buffer Triangles { vec4 centers[]; };
//   void main() {
//       uint columns = uint(ceil(sqrt(float(pc.count))));
//       vec2 cell = vec2(float(pc.index % columns), float(pc.index / columns));
//       centers[pc.index] =
//           vec4((cell + 0.5) / float(columns) * 2.0 - 1.0, 0.5, 1.0 / float(columns));
//   }
const uint32_t COMPUTE_SPIRV[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x00000044, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0005000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060010, 0x00000004, 0x00000011,
    0x00000001, 0x00000001, 0x00000001, 0x00030047, 0x00000009, 0x00000002, 0x00050048, 0x00000009,
    0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000009, 0x00000001, 0x00000023, 0x00000004,
    0x00040047, 0x00000026, 0x00000006, 0x00000010, 0x00030047, 0x00000027, 0x00000003, 0x00050048,
    0x00000027, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000029, 0x00000021, 0x00000000,
    0x00040047, 0x00000029, 0x00000022, 0x00000000, 0x00040047, 0x00000043, 0x0000000b, 0x00000019,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
    0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x0004001e, 0x00000009, 0x00000006,
    0x00000006, 0x00040020, 0x0000000a, 0x00000009, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b,
    0x00000009, 0x00040015, 0x0000000c, 0x00000020, 0x00000001, 0x0004002b, 0x0000000c, 0x0000000d,
    0x00000001, 0x00040020, 0x0000000e, 0x00000009, 0x00000006, 0x00030016, 0x00000011, 0x00000020,
    0x00040017, 0x00000016, 0x00000011, 0x00000002, 0x00040020, 0x00000017, 0x00000007, 0x00000016,
    0x0004002b, 0x0000000c, 0x00000019, 0x00000000, 0x00040017, 0x00000025, 0x00000011, 0x00000004,
    0x0003001d, 0x00000026, 0x00000025, 0x0003001e, 0x00000027, 0x00000026, 0x00040020, 0x00000028,
    0x00000002, 0x00000027, 0x0004003b, 0x00000028, 0x00000029, 0x00000002, 0x0004002b, 0x00000011,
    0x0000002d, 0x3f000000, 0x0004002b, 0x00000011, 0x00000034, 0x40000000, 0x0004002b, 0x00000011,
    0x00000036, 0x3f800000, 0x00040020, 0x0000003f, 0x00000002, 0x00000025, 0x00040017, 0x00000041,
    0x00000006, 0x00000003, 0x0004002b, 0x00000006, 0x00000042, 0x00000001, 0x0006002c, 0x00000041,
    0x00000043, 0x00000042, 0x00000042, 0x00000042, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x0004003b,
    0x00000017, 0x00000018, 0x00000007, 0x00050041, 0x0000000e, 0x0000000f, 0x0000000b, 0x0000000d,
    0x0004003d, 0x00000006, 0x00000010, 0x0000000f, 0x00040070, 0x00000011, 0x00000012, 0x00000010,
    0x0006000c, 0x00000011, 0x00000013, 0x00000001, 0x0000001f, 0x00000012, 0x0006000c, 0x00000011,
    0x00000014, 0x00000001, 0x00000009, 0x00000013, 0x0004006d, 0x00000006, 0x00000015, 0x00000014,
    0x0003003e, 0x00000008, 0x00000015, 0x00050041, 0x0000000e, 0x0000001a, 0x0000000b, 0x00000019,
    0x0004003d, 0x00000006, 0x0000001b, 0x0000001a, 0x0004003d, 0x00000006, 0x0000001c, 0x00000008,
    0x00050089, 0x00000006, 0x0000001d, 0x0000001b, 0x0000001c, 0x00040070, 0x00000011, 0x0000001e,
    0x0000001d, 0x00050041, 0x0000000e, 0x0000001f, 0x0000000b, 0x00000019, 0x0004003d, 0x00000006,
    0x00000020, 0x0000001f, 0x0004003d, 0x00000006, 0x00000021, 0x00000008, 0x00050086, 0x00000006,
    0x00000022, 0x00000020, 0x00000021, 0x00040070, 0x00000011, 0x00000023, 0x00000022, 0x00050050,
    0x00000016, 0x00000024, 0x0000001e, 0x00000023, 0x0003003e, 0x00000018, 0x00000024, 0x00050041,
    0x0000000e, 0x0000002a, 0x0000000b, 0x00000019, 0x0004003d, 0x00000006, 0x0000002b, 0x0000002a,
    0x0004003d, 0x00000016, 0x0000002c, 0x00000018, 0x00050050, 0x00000016, 0x0000002e, 0x0000002d,
    0x0000002d, 0x00050081, 0x00000016, 0x0000002f, 0x0000002c, 0x0000002e, 0x0004003d, 0x00000006,
    0x00000030, 0x00000008, 0x00040070, 0x00000011, 0x00000031, 0x00000030, 0x00050050, 0x00000016,
    0x00000032, 0x00000031, 0x00000031, 0x00050088, 0x00000016, 0x00000033, 0x0000002f, 0x00000032,
    0x0005008e, 0x00000016, 0x00000035, 0x00000033, 0x00000034, 0x00050050, 0x00000016, 0x00000037,
    0x00000036, 0x00000036, 0x00050083, 0x00000016, 0x00000038, 0x00000035, 0x00000037, 0x0004003d,
    0x00000006, 0x00000039, 0x00000008, 0x00040070, 0x00000011, 0x0000003a, 0x00000039, 0x00050088,
    0x00000011, 0x0000003b, 0x00000036, 0x0000003a, 0x00050051, 0x00000011, 0x0000003c, 0x00000038,
    0x00000000, 0x00050051, 0x00000011, 0x0000003d, 0x00000038, 0x00000001, 0x00070050, 0x00000025,
    0x0000003e, 0x0000003c, 0x0000003d, 0x0000002d, 0x0000003b, 0x00060041, 0x0000003f, 0x00000040,
    0x00000029, 0x00000019, 0x0000002b, 0x0003003e, 0x00000040, 0x0000003e, 0x000100fd, 0x00010038,
};

// probe.vert: a small triangle at centers[index], as mesh emulation's draw reads its output.
//   (same Constants and Triangles, the buffer readonly)
//   void main() {
//       vec4 center = centers[pc.index];
//       vec2 corner = vec2(float(gl_VertexIndex == 1), float(gl_VertexIndex == 2)) - 0.33;
//       gl_Position = vec4(center.xy + corner * center.w, center.z, 1.0);
//   }
const uint32_t VERTEX_SPIRV[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x00000044, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000001e, 0x00000032, 0x00040047,
    0x0000000a, 0x00000006, 0x00000010, 0x00030047, 0x0000000b, 0x00000003, 0x00040048, 0x0000000b,
    0x00000000, 0x00000018, 0x00050048, 0x0000000b, 0x00000000, 0x00000023, 0x00000000, 0x00030047,
    0x0000000d, 0x00000018, 0x00040047, 0x0000000d, 0x00000021, 0x00000000, 0x00040047, 0x0000000d,
    0x00000022, 0x00000000, 0x00030047, 0x00000011, 0x00000002, 0x00050048, 0x00000011, 0x00000000,
    0x00000023, 0x00000000, 0x00050048, 0x00000011, 0x00000001, 0x00000023, 0x00000004, 0x00040047,
    0x0000001e, 0x0000000b, 0x0000002a, 0x00030047, 0x00000030, 0x00000002, 0x00050048, 0x00000030,
    0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x00000030, 0x00000001, 0x0000000b, 0x00000001,
    0x00050048, 0x00000030, 0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x00000030, 0x00000003,
    0x0000000b, 0x00000004, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
    0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008,
    0x00000007, 0x00000007, 0x0003001d, 0x0000000a, 0x00000007, 0x0003001e, 0x0000000b, 0x0000000a,
    0x00040020, 0x0000000c, 0x00000002, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000002,
    0x00040015, 0x0000000e, 0x00000020, 0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000,
    0x00040015, 0x00000010, 0x00000020, 0x00000000, 0x0004001e, 0x00000011, 0x00000010, 0x00000010,
    0x00040020, 0x00000012, 0x00000009, 0x00000011, 0x0004003b, 0x00000012, 0x00000013, 0x00000009,
    0x00040020, 0x00000014, 0x00000009, 0x00000010, 0x00040020, 0x00000017, 0x00000002, 0x00000007,
    0x00040017, 0x0000001a, 0x00000006, 0x00000002, 0x00040020, 0x0000001b, 0x00000007, 0x0000001a,
    0x00040020, 0x0000001d, 0x00000001, 0x0000000e, 0x0004003b, 0x0000001d, 0x0000001e, 0x00000001,
    0x0004002b, 0x0000000e, 0x00000020, 0x00000001, 0x00020014, 0x00000021, 0x0004002b, 0x00000006,
    0x00000023, 0x00000000, 0x0004002b, 0x00000006, 0x00000024, 0x3f800000, 0x0004002b, 0x0000000e,
    0x00000027, 0x00000002, 0x0004002b, 0x00000006, 0x0000002b, 0x3ea8f5c3, 0x0004002b, 0x00000010,
    0x0000002e, 0x00000001, 0x0004001c, 0x0000002f, 0x00000006, 0x0000002e, 0x0006001e, 0x00000030,
    0x00000007, 0x00000006, 0x0000002f, 0x0000002f, 0x00040020, 0x00000031, 0x00000003, 0x00000030,
    0x0004003b, 0x00000031, 0x00000032, 0x00000003, 0x0004002b, 0x00000010, 0x00000036, 0x00000003,
    0x00040020, 0x00000037, 0x00000007, 0x00000006, 0x0004002b, 0x00000010, 0x0000003c, 0x00000002,
    0x00040020, 0x00000042, 0x00000003, 0x00000007, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000008, 0x00000009, 0x00000007, 0x0004003b,
    0x0000001b, 0x0000001c, 0x00000007, 0x00050041, 0x00000014, 0x00000015, 0x00000013, 0x0000000f,
    0x0004003d, 0x00000010, 0x00000016, 0x00000015, 0x00060041, 0x00000017, 0x00000018, 0x0000000d,
    0x0000000f, 0x00000016, 0x0004003d, 0x00000007, 0x00000019, 0x00000018, 0x0003003e, 0x00000009,
    0x00000019, 0x0004003d, 0x0000000e, 0x0000001f, 0x0000001e, 0x000500aa, 0x00000021, 0x00000022,
    0x0000001f, 0x00000020, 0x000600a9, 0x00000006, 0x00000025, 0x00000022, 0x00000024, 0x00000023,
    0x0004003d, 0x0000000e, 0x00000026, 0x0000001e, 0x000500aa, 0x00000021, 0x00000028, 0x00000026,
    0x00000027, 0x000600a9, 0x00000006, 0x00000029, 0x00000028, 0x00000024, 0x00000023, 0x00050050,
    0x0000001a, 0x0000002a, 0x00000025, 0x00000029, 0x00050050, 0x0000001a, 0x0000002c, 0x0000002b,
    0x0000002b, 0x00050083, 0x0000001a, 0x0000002d, 0x0000002a, 0x0000002c, 0x0003003e, 0x0000001c,
    0x0000002d, 0x0004003d, 0x00000007, 0x00000033, 0x00000009, 0x0007004f, 0x0000001a, 0x00000034,
    0x00000033, 0x00000033, 0x00000000, 0x00000001, 0x0004003d, 0x0000001a, 0x00000035, 0x0000001c,
    0x00050041, 0x00000037, 0x00000038, 0x00000009, 0x00000036, 0x0004003d, 0x00000006, 0x00000039,
    0x00000038, 0x0005008e, 0x0000001a, 0x0000003a, 0x00000035, 0x00000039, 0x00050081, 0x0000001a,
    0x0000003b, 0x00000034, 0x0000003a, 0x00050041, 0x00000037, 0x0000003d, 0x00000009, 0x0000003c,
    0x0004003d, 0x00000006, 0x0000003e, 0x0000003d, 0x00050051, 0x00000006, 0x0000003f, 0x0000003b,
    0x00000000, 0x00050051, 0x00000006, 0x00000040, 0x0000003b, 0x00000001, 0x00070050, 0x00000007,
    0x00000041, 0x0000003f, 0x00000040, 0x0000003e, 0x00000024, 0x00050041, 0x00000042, 0x00000043,
    0x00000032, 0x0000000f, 0x0003003e, 0x00000043, 0x00000041, 0x000100fd, 0x00010038,
};

// probe.frag: a colour from `index` to locations 0-3.
//   vec4 color = vec4(float(pc.index % 7u + 1u) / 8.0, float(pc.index % 5u + 1u) / 6.0, 0.25, 1.0);
const uint32_t FRAGMENT_SPIRV[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x0000002e, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0009000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000026, 0x00000028, 0x0000002a,
    0x0000002c, 0x00030010, 0x00000004, 0x00000007, 0x00030047, 0x0000000b, 0x00000002, 0x00050048,
    0x0000000b, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000000b, 0x00000001, 0x00000023,
    0x00000004, 0x00040047, 0x00000026, 0x0000001e, 0x00000000, 0x00040047, 0x00000028, 0x0000001e,
    0x00000001, 0x00040047, 0x0000002a, 0x0000001e, 0x00000002, 0x00040047, 0x0000002c, 0x0000001e,
    0x00000003, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
    0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000007,
    0x00000007, 0x00040015, 0x0000000a, 0x00000020, 0x00000000, 0x0004001e, 0x0000000b, 0x0000000a,
    0x0000000a, 0x00040020, 0x0000000c, 0x00000009, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d,
    0x00000009, 0x00040015, 0x0000000e, 0x00000020, 0x00000001, 0x0004002b, 0x0000000e, 0x0000000f,
    0x00000000, 0x00040020, 0x00000010, 0x00000009, 0x0000000a, 0x0004002b, 0x0000000a, 0x00000013,
    0x00000007, 0x0004002b, 0x0000000a, 0x00000015, 0x00000001, 0x0004002b, 0x00000006, 0x00000018,
    0x41000000, 0x0004002b, 0x0000000a, 0x0000001c, 0x00000005, 0x0004002b, 0x00000006, 0x00000020,
    0x40c00000, 0x0004002b, 0x00000006, 0x00000022, 0x3e800000, 0x0004002b, 0x00000006, 0x00000023,
    0x3f800000, 0x00040020, 0x00000025, 0x00000003, 0x00000007, 0x0004003b, 0x00000025, 0x00000026,
    0x00000003, 0x0004003b, 0x00000025, 0x00000028, 0x00000003, 0x0004003b, 0x00000025, 0x0000002a,
    0x00000003, 0x0004003b, 0x00000025, 0x0000002c, 0x00000003, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000008, 0x00000009, 0x00000007,
    0x00050041, 0x00000010, 0x00000011, 0x0000000d, 0x0000000f, 0x0004003d, 0x0000000a, 0x00000012,
    0x00000011, 0x00050089, 0x0000000a, 0x00000014, 0x00000012, 0x00000013, 0x00050080, 0x0000000a,
    0x00000016, 0x00000014, 0x00000015, 0x00040070, 0x00000006, 0x00000017, 0x00000016, 0x00050088,
    0x00000006, 0x00000019, 0x00000017, 0x00000018, 0x00050041, 0x00000010, 0x0000001a, 0x0000000d,
    0x0000000f, 0x0004003d, 0x0000000a, 0x0000001b, 0x0000001a, 0x00050089, 0x0000000a, 0x0000001d,
    0x0000001b, 0x0000001c, 0x00050080, 0x0000000a, 0x0000001e, 0x0000001d, 0x00000015, 0x00040070,
    0x00000006, 0x0000001f, 0x0000001e, 0x00050088, 0x00000006, 0x00000021, 0x0000001f, 0x00000020,
    0x00070050, 0x00000007, 0x00000024, 0x00000019, 0x00000021, 0x00000022, 0x00000023, 0x0003003e,
    0x00000009, 0x00000024, 0x0004003d, 0x00000007, 0x00000027, 0x00000009, 0x0003003e, 0x00000026,
    0x00000027, 0x0004003d, 0x00000007, 0x00000029, 0x00000009, 0x0003003e, 0x00000028, 0x00000029,
    0x0004003d, 0x00000007, 0x0000002b, 0x00000009, 0x0003003e, 0x0000002a, 0x0000002b, 0x0004003d,
    0x00000007, 0x0000002d, 0x00000009, 0x0003003e, 0x0000002c, 0x0000002d, 0x000100fd, 0x00010038,
};

#define VK_FUNCTIONS(X)                                                                            \
    X(CreateInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties)                   \
    X(GetPhysicalDeviceProperties2) X(GetPhysicalDeviceMemoryProperties)                           \
    X(GetPhysicalDeviceQueueFamilyProperties) X(CreateDevice) X(GetDeviceQueue)                    \
    X(CreateCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer) X(EndCommandBuffer)       \
    X(QueueSubmit) X(CreateFence) X(WaitForFences) X(ResetFences) X(CreateImage)                   \
    X(GetImageMemoryRequirements) X(AllocateMemory) X(BindImageMemory) X(CreateImageView)          \
    X(CreateRenderPass) X(CreateFramebuffer) X(CreateShaderModule) X(CreatePipelineLayout)         \
    X(CreateGraphicsPipelines) X(CreateComputePipelines) X(CreateDescriptorSetLayout)              \
    X(CreateDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CreateBuffer)      \
    X(GetBufferMemoryRequirements) X(BindBufferMemory) X(MapMemory) X(CmdBeginRenderPass)          \
    X(CmdEndRenderPass) X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdPushConstants)            \
    X(CmdDraw) X(CmdDispatch) X(CmdPipelineBarrier) X(CmdCopyImageToBuffer) X(DeviceWaitIdle)

struct Functions {
#define X(name) PFN_vk##name name{};
    VK_FUNCTIONS(X)
#undef X
} vk;

[[noreturn]] void Fail(const char* what, VkResult result = VK_SUCCESS) {
    std::fprintf(stderr, "%s failed (%d)\n", what, static_cast<int>(result));
    std::exit(1);
}

void Check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        Fail(what, result);
    }
}

PFN_vkGetInstanceProcAddr LoadEntry(const char* icd) {
    void* const library = dlopen(icd ? icd : "libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!library && !icd) {
        if (void* const versioned = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL)) {
            return reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                dlsym(versioned, "vkGetInstanceProcAddr"));
        }
    }
    if (!library) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        std::exit(1);
    }
    if (!icd) {
        return reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
    }
    using Negotiate = VkResult (*)(uint32_t*);
    if (const auto negotiate = reinterpret_cast<Negotiate>(
            dlsym(library, "vk_icdNegotiateLoaderICDInterfaceVersion"))) {
        uint32_t version = 5;
        negotiate(&version);
    }
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vk_icdGetInstanceProcAddr"));
}

struct Options {
    uint32_t width = 1920;
    uint32_t height = 1080;
    uint32_t colors = 3;
    uint32_t draws = 64;
    uint32_t frames = 20;
    bool half_float = false;
    const char* icd = nullptr;
};

Options Parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg{argv[i]};
        const auto value = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", arg.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        const auto number = [&](uint32_t low, uint32_t high) {
            return std::clamp(static_cast<uint32_t>(std::strtoul(value(), nullptr, 10)), low, high);
        };
        if (arg == "--width") {
            options.width = number(16, 8192);
        } else if (arg == "--height") {
            options.height = number(16, 8192);
        } else if (arg == "--colors") {
            options.colors = number(1, 4);
        } else if (arg == "--draws") {
            options.draws = number(1, 4096);
        } else if (arg == "--frames") {
            options.frames = number(1, 1000);
        } else if (arg == "--half-float") {
            options.half_float = true;
        } else if (arg == "--icd") {
            options.icd = value();
        } else {
            std::fprintf(stderr,
                         "usage: tile_break_probe [--width 1920] [--height 1080] [--colors 3] "
                         "[--draws 64]\n       [--frames 20] [--half-float] [--icd <driver .so>]\n");
            std::exit(2);
        }
    }
    return options;
}

struct Context {
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkPhysicalDeviceMemoryProperties memory{};

    VkDeviceMemory Allocate(const VkMemoryRequirements& requirements,
                            VkMemoryPropertyFlags wanted) const {
        for (uint32_t pass = 0; pass < 2; ++pass) {
            for (uint32_t type = 0; type < memory.memoryTypeCount; ++type) {
                const VkMemoryPropertyFlags flags = memory.memoryTypes[type].propertyFlags;
                if ((requirements.memoryTypeBits & (1U << type)) == 0 ||
                    (pass == 0 && (flags & wanted) != wanted)) {
                    continue;
                }
                const VkMemoryAllocateInfo info{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .allocationSize = requirements.size,
                    .memoryTypeIndex = type,
                };
                VkDeviceMemory allocation{};
                if (vk.AllocateMemory(device, &info, nullptr, &allocation) == VK_SUCCESS) {
                    return allocation;
                }
            }
            if (wanted & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
                break;
            }
        }
        Fail("vkAllocateMemory");
    }

    VkImageView Attachment(VkFormat format, VkExtent2D extent, VkImageUsageFlags usage,
                           VkImageAspectFlags aspect, VkImage& image) const {
        const VkImageCreateInfo image_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {extent.width, extent.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = usage,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        Check(vk.CreateImage(device, &image_info, nullptr, &image), "vkCreateImage");
        VkMemoryRequirements requirements{};
        vk.GetImageMemoryRequirements(device, image, &requirements);
        Check(vk.BindImageMemory(device, image,
                                 Allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0),
              "vkBindImageMemory");
        const VkImageViewCreateInfo view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .subresourceRange = {aspect, 0, 1, 0, 1},
        };
        VkImageView view{};
        Check(vk.CreateImageView(device, &view_info, nullptr, &view), "vkCreateImageView");
        return view;
    }

    VkBuffer Buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags wanted,
                    void** mapped = nullptr) const {
        const VkBufferCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
        };
        VkBuffer buffer{};
        Check(vk.CreateBuffer(device, &info, nullptr, &buffer), "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        vk.GetBufferMemoryRequirements(device, buffer, &requirements);
        const VkDeviceMemory allocation = Allocate(requirements, wanted);
        Check(vk.BindBufferMemory(device, buffer, allocation, 0), "vkBindBufferMemory");
        if (mapped) {
            Check(vk.MapMemory(device, allocation, 0, VK_WHOLE_SIZE, 0, mapped), "vkMapMemory");
        }
        return buffer;
    }

    VkShaderModule Shader(const uint32_t* words, size_t bytes) const {
        const VkShaderModuleCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = bytes,
            .pCode = words,
        };
        VkShaderModule shader{};
        Check(vk.CreateShaderModule(device, &info, nullptr, &shader), "vkCreateShaderModule");
        return shader;
    }
};

VkRenderPass MakeRenderPass(const Context& context, VkFormat color_format, VkFormat depth_format,
                            uint32_t colors, VkAttachmentLoadOp load) {
    std::vector<VkAttachmentDescription> attachments;
    std::vector<VkAttachmentReference> references;
    for (uint32_t i = 0; i <= colors; ++i) {
        // Everything stays in the general layout, as in the emulator.
        attachments.push_back({
            .format = i < colors ? color_format : depth_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = load,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
            .finalLayout = VK_IMAGE_LAYOUT_GENERAL,
        });
        references.push_back({i, VK_IMAGE_LAYOUT_GENERAL});
    }
    const VkSubpassDescription subpass{
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = colors,
        .pColorAttachments = references.data(),
        .pDepthStencilAttachment = &references[colors],
    };
    const VkRenderPassCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = colors + 1,
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
    };
    VkRenderPass render_pass{};
    Check(vk.CreateRenderPass(context.device, &info, nullptr, &render_pass), "vkCreateRenderPass");
    return render_pass;
}

enum class Mode { DispatchesFirst, BreakPerDraw, NoDispatches };

} // Anonymous namespace

int main(int argc, char** argv) {
    const Options options = Parse(argc, argv);
    const PFN_vkGetInstanceProcAddr entry = LoadEntry(options.icd);
    if (!entry) {
        std::fprintf(stderr, "no Vulkan entry point in %s\n", options.icd ? options.icd : "libvulkan.so");
        return 1;
    }
    vk.CreateInstance = reinterpret_cast<PFN_vkCreateInstance>(entry(nullptr, "vkCreateInstance"));
    const VkApplicationInfo application{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "tile_break_probe",
        .apiVersion = VK_API_VERSION_1_1,
    };
    const VkInstanceCreateInfo instance_info{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &application,
    };
    VkInstance instance{};
    Check(vk.CreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");
#define X(name)                                                                                    \
    if (!vk.name) {                                                                                \
        vk.name = reinterpret_cast<PFN_vk##name>(entry(instance, "vk" #name));                     \
    }
    VK_FUNCTIONS(X)
#undef X

    Context context;
    uint32_t count = 1;
    const VkResult enumerated = vk.EnumeratePhysicalDevices(instance, &count, &context.physical);
    if ((enumerated != VK_SUCCESS && enumerated != VK_INCOMPLETE) || count == 0) {
        Fail("vkEnumeratePhysicalDevices", enumerated);
    }
    VkPhysicalDeviceProperties properties{};
    vk.GetPhysicalDeviceProperties(context.physical, &properties);
    VkPhysicalDeviceDriverProperties driver{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
    };
    if (properties.apiVersion >= VK_API_VERSION_1_1 && vk.GetPhysicalDeviceProperties2) {
        VkPhysicalDeviceProperties2 properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &driver,
        };
        vk.GetPhysicalDeviceProperties2(context.physical, &properties2);
    }
    vk.GetPhysicalDeviceMemoryProperties(context.physical, &context.memory);

    uint32_t family_count = 0;
    vk.GetPhysicalDeviceQueueFamilyProperties(context.physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vk.GetPhysicalDeviceQueueFamilyProperties(context.physical, &family_count, families.data());
    uint32_t family = 0;
    constexpr VkQueueFlags wanted_queue = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    while (family < family_count && (families[family].queueFlags & wanted_queue) != wanted_queue) {
        ++family;
    }
    if (family == family_count) {
        Fail("finding a graphics and compute queue");
    }
    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    const VkDeviceCreateInfo device_info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
    };
    Check(vk.CreateDevice(context.physical, &device_info, nullptr, &context.device),
          "vkCreateDevice");
    vk.GetDeviceQueue(context.device, family, 0, &context.queue);

    const VkExtent2D extent{options.width, options.height};
    const VkFormat color_format =
        options.half_float ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkFormat depth_format = VK_FORMAT_D32_SFLOAT;
    const uint32_t colors = options.colors;
    const uint32_t draws = options.draws;

    std::vector<VkImage> images(colors + 1);
    std::vector<VkImageView> views(colors + 1);
    for (uint32_t i = 0; i < colors; ++i) {
        views[i] = context.Attachment(
            color_format, extent,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, images[i]);
    }
    views[colors] = context.Attachment(
        depth_format, extent,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT, images[colors]);

    const VkRenderPass clear_pass =
        MakeRenderPass(context, color_format, depth_format, colors, VK_ATTACHMENT_LOAD_OP_CLEAR);
    const VkRenderPass load_pass =
        MakeRenderPass(context, color_format, depth_format, colors, VK_ATTACHMENT_LOAD_OP_LOAD);
    const VkFramebufferCreateInfo framebuffer_info{
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = clear_pass,
        .attachmentCount = colors + 1,
        .pAttachments = views.data(),
        .width = extent.width,
        .height = extent.height,
        .layers = 1,
    };
    VkFramebuffer framebuffer{};
    Check(vk.CreateFramebuffer(context.device, &framebuffer_info, nullptr, &framebuffer),
          "vkCreateFramebuffer");

    const VkDescriptorSetLayoutBinding binding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
    };
    const VkDescriptorSetLayoutCreateInfo set_layout_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    VkDescriptorSetLayout set_layout{};
    Check(vk.CreateDescriptorSetLayout(context.device, &set_layout_info, nullptr, &set_layout),
          "vkCreateDescriptorSetLayout");
    constexpr VkShaderStageFlags push_stages =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    const VkPushConstantRange push_range{push_stages, 0, 8};
    const VkPipelineLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VkPipelineLayout layout{};
    Check(vk.CreatePipelineLayout(context.device, &layout_info, nullptr, &layout),
          "vkCreatePipelineLayout");

    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    const VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VkDescriptorPool descriptor_pool{};
    Check(vk.CreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool),
          "vkCreateDescriptorPool");
    const VkDescriptorSetAllocateInfo set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &set_layout,
    };
    VkDescriptorSet descriptor_set{};
    Check(vk.AllocateDescriptorSets(context.device, &set_info, &descriptor_set),
          "vkAllocateDescriptorSets");
    const VkBuffer triangles = context.Buffer(VkDeviceSize{draws} * 16,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const VkDescriptorBufferInfo buffer_info{triangles, 0, VK_WHOLE_SIZE};
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_info,
    };
    vk.UpdateDescriptorSets(context.device, 1, &write, 0, nullptr);

    const VkComputePipelineCreateInfo compute_info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = context.Shader(COMPUTE_SPIRV, sizeof(COMPUTE_SPIRV)),
            .pName = "main",
        },
        .layout = layout,
    };
    VkPipeline compute_pipeline{};
    Check(vk.CreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &compute_info, nullptr,
                                    &compute_pipeline),
          "vkCreateComputePipelines");

    const VkPipelineShaderStageCreateInfo stages[2]{
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = context.Shader(VERTEX_SPIRV, sizeof(VERTEX_SPIRV)),
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = context.Shader(FRAGMENT_SPIRV, sizeof(FRAGMENT_SPIRV)),
            .pName = "main",
        },
    };
    const VkPipelineVertexInputStateCreateInfo vertex_input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    const VkPipelineInputAssemblyStateCreateInfo input_assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const VkViewport viewport{0, 0, static_cast<float>(extent.width),
                              static_cast<float>(extent.height), 0, 1};
    const VkRect2D scissor{{0, 0}, extent};
    const VkPipelineViewportStateCreateInfo viewport_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };
    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const VkPipelineDepthStencilStateCreateInfo depth_stencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
    };
    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments(
        colors, VkPipelineColorBlendAttachmentState{.colorWriteMask = 0xf});
    const VkPipelineColorBlendStateCreateInfo blend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = colors,
        .pAttachments = blend_attachments.data(),
    };
    const VkGraphicsPipelineCreateInfo graphics_info{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &blend,
        .layout = layout,
        .renderPass = clear_pass,
    };
    VkPipeline graphics_pipeline{};
    Check(vk.CreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &graphics_info, nullptr,
                                     &graphics_pipeline),
          "vkCreateGraphicsPipelines");

    const VkDeviceSize readback_size = VkDeviceSize{extent.width} * extent.height *
                                       (options.half_float ? 8 : 4);
    void* readback_data{};
    const VkBuffer readback = context.Buffer(
        readback_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &readback_data);

    const VkCommandPoolCreateInfo command_pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = family,
    };
    VkCommandPool command_pool{};
    Check(vk.CreateCommandPool(context.device, &command_pool_info, nullptr, &command_pool),
          "vkCreateCommandPool");
    const VkCommandBufferAllocateInfo command_buffer_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmdbuf{};
    Check(vk.AllocateCommandBuffers(context.device, &command_buffer_info, &cmdbuf),
          "vkAllocateCommandBuffers");
    const VkFenceCreateInfo fence_info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence{};
    Check(vk.CreateFence(context.device, &fence_info, nullptr, &fence), "vkCreateFence");

    // Records with `record`, submits and returns how long the GPU took, in milliseconds.
    const auto run = [&](auto&& record) {
        const VkCommandBufferBeginInfo begin{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        Check(vk.BeginCommandBuffer(cmdbuf, &begin), "vkBeginCommandBuffer");
        record();
        Check(vk.EndCommandBuffer(cmdbuf), "vkEndCommandBuffer");
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmdbuf,
        };
        const auto start = std::chrono::steady_clock::now();
        Check(vk.QueueSubmit(context.queue, 1, &submit, fence), "vkQueueSubmit");
        Check(vk.WaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        const auto end = std::chrono::steady_clock::now();
        Check(vk.ResetFences(context.device, 1, &fence), "vkResetFences");
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    run([&] {
        std::vector<VkImageMemoryBarrier> barriers;
        for (uint32_t i = 0; i <= colors; ++i) {
            barriers.push_back({
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = images[i],
                .subresourceRange = {i < colors ? VkImageAspectFlags{VK_IMAGE_ASPECT_COLOR_BIT}
                                                : VkImageAspectFlags{VK_IMAGE_ASPECT_DEPTH_BIT},
                                     0, 1, 0, 1},
            });
        }
        vk.CmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                              static_cast<uint32_t>(barriers.size()), barriers.data());
    });

    const auto record_frame = [&](Mode mode) {
        std::vector<VkClearValue> clears(colors + 1);
        clears[colors].depthStencil = {1.0f, 0};
        bool in_pass = false;
        bool cleared = false;
        const auto begin_pass = [&] {
            const VkRenderPassBeginInfo info{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = cleared ? load_pass : clear_pass,
                .framebuffer = framebuffer,
                .renderArea = {{0, 0}, extent},
                .clearValueCount = cleared ? 0 : colors + 1,
                .pClearValues = cleared ? nullptr : clears.data(),
            };
            vk.CmdBeginRenderPass(cmdbuf, &info, VK_SUBPASS_CONTENTS_INLINE);
            vk.CmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
            vk.CmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1,
                                     &descriptor_set, 0, nullptr);
            in_pass = true;
            cleared = true;
        };
        const auto end_pass = [&] {
            if (std::exchange(in_pass, false)) {
                vk.CmdEndRenderPass(cmdbuf);
            }
        };
        const auto dispatch = [&](uint32_t index) {
            const uint32_t constants[2]{index, draws};
            vk.CmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
            vk.CmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                                     &descriptor_set, 0, nullptr);
            vk.CmdPushConstants(cmdbuf, layout, push_stages, 0, sizeof(constants), constants);
            vk.CmdDispatch(cmdbuf, 1, 1, 1);
            const VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            };
            vk.CmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        };
        if (mode == Mode::DispatchesFirst) {
            for (uint32_t index = 0; index < draws; ++index) {
                dispatch(index);
            }
        }
        for (uint32_t index = 0; index < draws; ++index) {
            if (mode == Mode::BreakPerDraw) {
                end_pass();
                dispatch(index);
            }
            if (!in_pass) {
                begin_pass();
            }
            const uint32_t constants[2]{index, draws};
            vk.CmdPushConstants(cmdbuf, layout, push_stages, 0, sizeof(constants), constants);
            vk.CmdDraw(cmdbuf, 3, 1, 0, 0);
        }
        end_pass();
    };

    const auto checksum = [&] {
        run([&] {
            const VkBufferImageCopy region{
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageExtent = {extent.width, extent.height, 1},
            };
            vk.CmdCopyImageToBuffer(cmdbuf, images[0], VK_IMAGE_LAYOUT_GENERAL, readback, 1,
                                    &region);
        });
        uint64_t hash = 0xcbf29ce484222325ULL;
        const auto* const bytes = static_cast<const unsigned char*>(readback_data);
        for (VkDeviceSize i = 0; i < readback_size; ++i) {
            hash = (hash ^ bytes[i]) * 0x100000001b3ULL;
        }
        return hash;
    };

    struct Result {
        double median{};
        double best{};
        uint64_t image{};
    };
    const auto measure = [&](Mode mode) {
        // The first run fills NoDispatches' buffer too: its draws read what the others wrote.
        std::vector<double> times;
        for (uint32_t frame = 0; frame < options.frames + 3; ++frame) {
            const double ms = run([&] { record_frame(mode); });
            if (frame >= 3) {
                times.push_back(ms);
            }
        }
        std::sort(times.begin(), times.end());
        return Result{times[times.size() / 2], times.front(), checksum()};
    };

    std::printf("%s, driver %s %s\n", properties.deviceName, driver.driverName, driver.driverInfo);
    std::printf("%ux%u, %u %s colour attachments and depth, %u draws a frame, %u frames\n",
                extent.width, extent.height, colors, options.half_float ? "RGBA16F" : "RGBA8",
                draws, options.frames);
    const Result first = measure(Mode::DispatchesFirst);
    const Result breaks = measure(Mode::BreakPerDraw);
    const Result plain = measure(Mode::NoDispatches);
    std::printf("dispatches before one pass: %9.3f ms a frame (best %.3f)\n", first.median,
                first.best);
    std::printf("pass break before each draw:%9.3f ms a frame (best %.3f)\n", breaks.median,
                breaks.best);
    std::printf("one pass, no dispatches:    %9.3f ms a frame (best %.3f)\n", plain.median,
                plain.best);
    const double per_break = draws > 1 ? (breaks.median - first.median) / (draws - 1) : 0.0;
    std::printf("a pass break costs %.3f ms: %u breaks add %.1f ms to a frame\n", per_break,
                draws - 1, breaks.median - first.median);
    const bool same = first.image == breaks.image && first.image == plain.image;
    std::printf("images %s (%016llx %016llx %016llx)\n", same ? "match" : "DIFFER",
                static_cast<unsigned long long>(first.image),
                static_cast<unsigned long long>(breaks.image),
                static_cast<unsigned long long>(plain.image));
    vk.DeviceWaitIdle(context.device);
    return same ? 0 : 1;
}
