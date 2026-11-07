// SPDX-FileCopyrightText: Copyright 2026 Lichii Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/textures/texture.h"

namespace VideoCommon::OcarinaUpscale {

// The Legend of Zelda: Ocarina of Time upscales its 1920x1080 scene to the 2560x1440 layout with a
// tensor-core network: nine matrix-multiply passes, then an output pass that combines their result
// with the scene. Where the network can't run (software HMMA refused on Qualcomm's driver, or
// seconds per frame on Turnip), the output pass is replaced by a bilinear upscale of the scene.
constexpr u64 ProgramId = 0x0400486024DEC000ULL;
/// NativeProgramKey of the output pass (OnePlus 12 log): grid 80x180 of 128 threads, 2560x1440.
constexpr u64 OutputProgramKey = 0x6093c9143ba0904dULL;
/// Its sampled 1920x1080 images: A8B8G8R8, and A2B10G10R10 (also written back through 0xc0).
constexpr u32 SceneColorOffset = 0xa8;
constexpr u32 FeedbackColorOffset = 0xa0;
/// Its 2560x1440 A8B8G8R8 storage image.
constexpr u32 OutputOffset = 0xc8;
constexpr u32 ConstantsSize = OutputOffset + sizeof(u32);

inline bool UnormColor(const Tegra::Texture::TICEntry& image, Tegra::Texture::TextureFormat format) {
    using namespace Tegra::Texture;
    return image.format == format && image.IsBlockLinear() && image.Address() != 0 &&
           image.r_type == ComponentType::UNORM && image.g_type == ComponentType::UNORM &&
           image.b_type == ComponentType::UNORM && image.a_type == ComponentType::UNORM;
}

inline bool ValidInput(const Tegra::Texture::TICEntry& image) {
    using Tegra::Texture::TextureFormat;
    return UnormColor(image, TextureFormat::A8B8G8R8) ||
           UnormColor(image, TextureFormat::A2B10G10R10);
}

inline bool ValidPair(const Tegra::Texture::TICEntry& input, const Tegra::Texture::TICEntry& output) {
    return ValidInput(input) && UnormColor(output, Tegra::Texture::TextureFormat::A8B8G8R8) &&
           input.Address() != output.Address() && output.Width() >= input.Width() &&
           output.Height() >= input.Height();
}

} // namespace VideoCommon::OcarinaUpscale
