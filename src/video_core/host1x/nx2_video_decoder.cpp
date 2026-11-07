// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <set>
#include <thread>

#ifdef EDEN_NX2_LIBGAV1
#include <cstdlib>
#include <gav1/decoder.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>
#endif

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/host1x/ffmpeg.h"
#include "video_core/host1x/h264_packet.h"
#include "video_core/host1x/nx2_video_decoder.h"

extern "C" {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif

#include <libswscale/swscale.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

namespace Tegra::Host1x {

namespace {

std::optional<NvdecCommon::VideoCodec> CodecFromMime(std::string_view mime) {
    if (mime.find("vp9") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::VP9;
    }
    if (mime.find("vp8") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::VP8;
    }
    // "video/av01" or "video/av1". FFmpeg decodes AV1 with libdav1d.
    if (mime.find("av01") != std::string_view::npos || mime.find("av1") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::AV1;
    }
    if (mime.find("avc") != std::string_view::npos || mime.find("264") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::H264;
    }
    return std::nullopt;
}

#ifdef EDEN_NX2_LIBGAV1
AVPixelFormat Av1PixelFormat(const libgav1::DecoderBuffer& buffer) {
    if (buffer.bitdepth != 8 && buffer.bitdepth != 10) {
        return AV_PIX_FMT_NONE;
    }
    const bool ten_bit = buffer.bitdepth == 10;
    switch (buffer.image_format) {
    case kLibgav1ImageFormatYuv420:
        return ten_bit ? AV_PIX_FMT_YUV420P10LE : AV_PIX_FMT_YUV420P;
    case kLibgav1ImageFormatYuv422:
        return ten_bit ? AV_PIX_FMT_YUV422P10LE : AV_PIX_FMT_YUV422P;
    case kLibgav1ImageFormatYuv444:
        return ten_bit ? AV_PIX_FMT_YUV444P10LE : AV_PIX_FMT_YUV444P;
    case kLibgav1ImageFormatMonochrome400:
        return ten_bit ? AV_PIX_FMT_GRAY10LE : AV_PIX_FMT_GRAY8;
    }
    return AV_PIX_FMT_NONE;
}

// Copies a picture libgav1 decoded into a frame of its own (libgav1 reuses its buffers), tagged
// with the stream's matrix and range for Render.
std::shared_ptr<FFmpeg::Frame> CopyAv1Picture(const libgav1::DecoderBuffer& buffer) {
    const AVPixelFormat format = Av1PixelFormat(buffer);
    const AVPixFmtDescriptor* descriptor = av_pix_fmt_desc_get(format);
    if (descriptor == nullptr || buffer.displayed_width[0] <= 0 ||
        buffer.displayed_height[0] <= 0) {
        return nullptr;
    }
    auto frame = std::make_shared<FFmpeg::Frame>();
    AVFrame* f = frame->GetFrame();
    f->format = format;
    f->width = buffer.displayed_width[0];
    f->height = buffer.displayed_height[0];
    if (av_frame_get_buffer(f, 0) < 0) {
        return nullptr;
    }
    const int sample_bytes = buffer.bitdepth > 8 ? 2 : 1;
    for (int plane = 0; plane < buffer.NumPlanes(); ++plane) {
        const int width =
            plane == 0 ? f->width : AV_CEIL_RSHIFT(f->width, descriptor->log2_chroma_w);
        const int height =
            plane == 0 ? f->height : AV_CEIL_RSHIFT(f->height, descriptor->log2_chroma_h);
        const size_t row_bytes =
            static_cast<size_t>(std::min(width, buffer.displayed_width[plane]) * sample_bytes);
        for (int y = 0; y < std::min(height, buffer.displayed_height[plane]); ++y) {
            std::memcpy(f->data[plane] + static_cast<ptrdiff_t>(y) * f->linesize[plane],
                        buffer.plane[plane] + static_cast<ptrdiff_t>(y) * buffer.stride[plane],
                        row_bytes);
        }
    }
    // AV1 numbers its matrices as FFmpeg does (ITU-T H.273).
    const int matrix = static_cast<int>(buffer.matrix_coefficients);
    f->colorspace = matrix <= static_cast<int>(AVCOL_SPC_ICTCP) ? static_cast<AVColorSpace>(matrix)
                                                                : AVCOL_SPC_UNSPECIFIED;
    f->color_range =
        buffer.color_range == kLibgav1ColorRangeFull ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
    return frame;
}

// Android's FFmpeg package has no AV1 MediaCodec decoder, and libgav1 decodes Ocarina of Time's
// 1080p60 movies at about 36 FPS on a Snapdragon 8 Gen 3 next to the emulated CPU, so the title
// backdrop stutters. Where the device has a hardware AV1 decoder, the NDK's MediaCodec decodes
// instead; libgav1 stays the fallback. EDEN_NX2_AV1_MEDIACODEC=0 keeps libgav1.
class MediaCodecAv1 {
public:
    static std::unique_ptr<MediaCodecAv1> Create() {
        if (const char* value = std::getenv("EDEN_NX2_AV1_MEDIACODEC"); value && value[0] == '0') {
            return nullptr;
        }
        AMediaCodec* codec = AMediaCodec_createDecoderByType("video/av01");
        if (codec == nullptr) {
            return nullptr;
        }
        AMediaFormat* format = AMediaFormat_new();
        AMediaFormat_setString(format, AMEDIAFORMAT_KEY_MIME, "video/av01");
        // A hint only: the decoder takes the size from the stream's sequence header.
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_WIDTH, 1920);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_HEIGHT, 1080);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_MAX_WIDTH, 3840);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_MAX_HEIGHT, 2160);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT, ColorYuv420Flexible);
        const bool started =
            AMediaCodec_configure(codec, format, nullptr, nullptr, 0) == AMEDIA_OK &&
            AMediaCodec_start(codec) == AMEDIA_OK;
        AMediaFormat_delete(format);
        if (!started) {
            AMediaCodec_delete(codec);
            return nullptr;
        }
        return std::unique_ptr<MediaCodecAv1>{new MediaCodecAv1{codec}};
    }

    ~MediaCodecAv1() {
        AMediaCodec_stop(codec);
        AMediaCodec_delete(codec);
    }

    std::string Name() const {
#if __ANDROID_API__ >= 28
        char* name{};
        if (AMediaCodec_getName(codec, &name) == AMEDIA_OK && name != nullptr) {
            std::string result{name};
            AMediaCodec_releaseName(codec, name);
            return result;
        }
#endif
        return "MediaCodec";
    }

    /// Queues one frame and takes the pictures that are finished. False if the decoder took no
    /// input for half a second or put out a layout this can't read.
    bool Decode(std::span<const u8> data, u64 pts, std::vector<Nx2Picture>& pictures) {
        if (!Queue(data, pts, 0, pictures)) {
            return false;
        }
        ++inputs;
        Receive(pictures, 0);
        return !unsupported;
    }

    /// End of stream: takes every picture still in the decoder, then starts over.
    void Drain(std::vector<Nx2Picture>& pictures) {
        if (Queue({}, 0, AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM, pictures)) {
            for (int wait = 0; wait < 100 && !Receive(pictures, 10000); ++wait) {
            }
        }
        Flush();
    }

    void Flush() {
        AMediaCodec_flush(codec);
    }

    /// Frames queued so far, and pictures put out: a decoder that takes the game's stream but
    /// never answers is given up on.
    u64 inputs{};
    u64 outputs{};

private:
    static constexpr s32 ColorYuv420Planar = 19;
    static constexpr s32 ColorYuv420SemiPlanar = 21;
    static constexpr s32 ColorYuvP010 = 54;
    static constexpr s32 ColorYuv420Flexible = 0x7F420888;
    static constexpr s32 ColorQcomYuv420SemiPlanar32m = 0x7FA30C04;

    explicit MediaCodecAv1(AMediaCodec* codec_) : codec{codec_} {}

    bool Queue(std::span<const u8> data, u64 pts, u32 flags, std::vector<Nx2Picture>& pictures) {
        for (int attempt = 0; attempt < 250; ++attempt) {
            const ssize_t index = AMediaCodec_dequeueInputBuffer(codec, 2000);
            if (index < 0) {
                // Every input buffer is in use: take finished pictures to free some.
                Receive(pictures, 0);
                continue;
            }
            size_t capacity{};
            u8* const buffer = AMediaCodec_getInputBuffer(codec, static_cast<size_t>(index), &capacity);
            if (buffer == nullptr || capacity < data.size()) {
                AMediaCodec_queueInputBuffer(codec, static_cast<size_t>(index), 0, 0, 0, 0);
                return false;
            }
            if (!data.empty()) {
                std::memcpy(buffer, data.data(), data.size());
            }
            return AMediaCodec_queueInputBuffer(codec, static_cast<size_t>(index), 0, data.size(),
                                                pts, flags) == AMEDIA_OK;
        }
        return false;
    }

    // Takes the finished pictures, waiting up to timeout_us for the first. True at end of stream.
    bool Receive(std::vector<Nx2Picture>& pictures, s64 timeout_us) {
        while (true) {
            AMediaCodecBufferInfo info{};
            const ssize_t index = AMediaCodec_dequeueOutputBuffer(codec, &info, timeout_us);
            if (index == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
                ReadFormat();
                continue;
            }
            if (index == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
                continue;
            }
            if (index < 0) {
                return false;
            }
            timeout_us = 0;
            const bool end = (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) != 0;
            size_t size{};
            const u8* const buffer =
                AMediaCodec_getOutputBuffer(codec, static_cast<size_t>(index), &size);
            if (buffer != nullptr && info.size > 0 &&
                static_cast<size_t>(info.offset) + static_cast<size_t>(info.size) <= size) {
                if (auto frame = CopyPicture(buffer + info.offset, static_cast<size_t>(info.size))) {
                    ++outputs;
                    pictures.push_back({width, height, static_cast<u64>(info.presentationTimeUs),
                                        std::move(frame)});
                }
            }
            AMediaCodec_releaseOutputBuffer(codec, static_cast<size_t>(index), false);
            if (end) {
                return true;
            }
        }
    }

    void ReadFormat() {
        AMediaFormat* format = AMediaCodec_getOutputFormat(codec);
        if (format == nullptr) {
            return;
        }
        s32 value{};
        s32 frame_width = AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_WIDTH, &value) ? value : 0;
        s32 frame_height = AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_HEIGHT, &value) ? value : 0;
        stride = AMediaFormat_getInt32(format, "stride", &value) ? value : frame_width;
        slice_height = AMediaFormat_getInt32(format, "slice-height", &value) ? value : frame_height;
        color_format = AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT, &value) ? value : 0;
        full_range = AMediaFormat_getInt32(format, "color-range", &value) && value == 1;
        s32 left{}, top{}, right{}, bottom{};
        if (AMediaFormat_getInt32(format, "crop-left", &left) &&
            AMediaFormat_getInt32(format, "crop-top", &top) &&
            AMediaFormat_getInt32(format, "crop-right", &right) &&
            AMediaFormat_getInt32(format, "crop-bottom", &bottom)) {
            crop_x = static_cast<u32>(std::max(left, 0));
            crop_y = static_cast<u32>(std::max(top, 0));
            frame_width = right - left + 1;
            frame_height = bottom - top + 1;
        }
        AMediaFormat_delete(format);
        width = static_cast<u32>(std::max(frame_width, 0));
        height = static_cast<u32>(std::max(frame_height, 0));
        stride = std::max(stride, frame_width + static_cast<s32>(crop_x));
        slice_height = std::max(slice_height, frame_height + static_cast<s32>(crop_y));
        unsupported = color_format != ColorYuv420Planar && color_format != ColorYuv420SemiPlanar &&
                      color_format != ColorQcomYuv420SemiPlanar32m && color_format != ColorYuvP010;
        LOG_INFO(HW_GPU, "nx2 video: hardware AV1 output {}x{} stride {} slice height {} crop {},{} "
                 "color format {:#x}{}", width, height, stride, slice_height, crop_x, crop_y,
                 color_format, unsupported ? " (unsupported)" : "");
    }

    // Copies the picture out of the codec's buffer (it is given back right after) as NV12, I420 or
    // P010, which Render converts like libgav1's.
    std::shared_ptr<FFmpeg::Frame> CopyPicture(const u8* data, size_t size) const {
        if (unsupported || width == 0 || height == 0) {
            return nullptr;
        }
        const bool p010 = color_format == ColorYuvP010;
        const bool planar = color_format == ColorYuv420Planar;
        const size_t sample = p010 ? 2 : 1;
        const size_t luma_size = static_cast<size_t>(stride) * static_cast<size_t>(slice_height);
        const size_t chroma_rows = (static_cast<size_t>(slice_height) + 1) / 2;
        const size_t needed = luma_size + (planar ? 2 * (stride / 2) * chroma_rows
                                                  : static_cast<size_t>(stride) * chroma_rows);
        if (size < needed) {
            return nullptr;
        }
        auto frame = std::make_shared<FFmpeg::Frame>();
        AVFrame* f = frame->GetFrame();
        f->format = p010 ? AV_PIX_FMT_P010LE : planar ? AV_PIX_FMT_YUV420P : AV_PIX_FMT_NV12;
        f->width = static_cast<int>(width);
        f->height = static_cast<int>(height);
        if (av_frame_get_buffer(f, 0) < 0) {
            return nullptr;
        }
        const auto copy = [&](int plane, size_t offset, size_t source_stride, u32 rows, u32 x,
                              u32 y, size_t row_bytes) {
            for (u32 row = 0; row < rows; ++row) {
                std::memcpy(f->data[plane] + static_cast<ptrdiff_t>(row) * f->linesize[plane],
                            data + offset + (y + row) * source_stride + x, row_bytes);
            }
        };
        const u32 chroma_height = (height + 1) / 2;
        copy(0, 0, static_cast<size_t>(stride), height, crop_x * sample, crop_y, width * sample);
        if (planar) {
            const size_t chroma_stride = static_cast<size_t>(stride) / 2;
            const size_t chroma_width = (width + 1) / 2;
            copy(1, luma_size, chroma_stride, chroma_height, crop_x / 2, crop_y / 2, chroma_width);
            copy(2, luma_size + chroma_stride * chroma_rows, chroma_stride, chroma_height,
                 crop_x / 2, crop_y / 2, chroma_width);
        } else {
            // Interleaved Cb/Cr rows (NV12, or 16-bit samples for P010).
            copy(1, luma_size, static_cast<size_t>(stride), chroma_height, (crop_x & ~1U) * sample,
                 crop_y / 2, ((width + 1) & ~1U) * sample);
        }
        f->color_range = full_range ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
        return frame;
    }

    AMediaCodec* codec{};
    u32 width{1920};
    u32 height{1080};
    s32 stride{1920};
    s32 slice_height{1080};
    u32 crop_x{};
    u32 crop_y{};
    s32 color_format{ColorYuv420SemiPlanar};
    bool full_range{};
    bool unsupported{};
};
#endif

} // namespace

bool IsH264Stream(std::span<const u8> data) {
    if (data.size() >= 4) {
        // Annex-B 3-byte start code: 00 00 01 <nal_header>
        if (data[0] == 0 && data[1] == 0 && data[2] == 1) {
            const u8 nal_type = data[3] & 0x1F;
            if ((data[3] & 0x80) == 0 && nal_type >= 1 && nal_type <= 23) {
                return true;
            }
        }
    }
    if (data.size() >= 5) {
        // Annex-B 4-byte start code: 00 00 00 01 <nal_header>
        if (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) {
            const u8 nal_type = data[4] & 0x1F;
            if ((data[4] & 0x80) == 0 && nal_type >= 1 && nal_type <= 23) {
                return true;
            }
        }
        // avcC / MP4 length-prefixed NAL unit: [u32 be length] <nal_header>
        const u32 nalu_len = (static_cast<u32>(data[0]) << 24) |
                             (static_cast<u32>(data[1]) << 16) |
                             (static_cast<u32>(data[2]) << 8) |
                             static_cast<u32>(data[3]);
        if (nalu_len > 0 && nalu_len <= data.size() - 4) {
            const u8 nal_type = data[4] & 0x1F;
            if ((data[4] & 0x80) == 0 && nal_type >= 1 && nal_type <= 23) {
                return true;
            }
        }
    }
    if (data.size() >= 7) {
        // avcC configuration record: version=1, profile, compat, level, lengthSizeMinusOne | 0xFC, numSPS | 0xE0
        if (data[0] == 1 && (data[4] & 0xFC) == 0xFC && (data[5] & 0xE0) == 0xE0 && (data[5] & 0x1F) > 0) {
            return true;
        }
    }
    return false;
}

bool Nx2IsAv1Mime(std::string_view mime) {
    return mime.find("av01") != std::string_view::npos ||
           mime.find("av1") != std::string_view::npos;
}

struct Nx2VideoDecoder::Impl {
    std::string name;
    std::optional<FFmpeg::Decoder> decoder;
    std::optional<FFmpeg::HardwareContext> hardware;
    std::optional<FFmpeg::DecoderContext> context;
#ifdef EDEN_NX2_LIBGAV1
    std::unique_ptr<MediaCodecAv1> hardware_av1; ///< Instead of libgav1 where the device has one
    std::unique_ptr<libgav1::Decoder> av1; ///< Instead of FFmpeg, for AV1
    u64 av1_errors{};

    bool OpenLibgav1() {
        auto decoder = std::make_unique<libgav1::Decoder>();
        libgav1::DecoderSettings settings;
        settings.threads =
            static_cast<int>(std::clamp(std::thread::hardware_concurrency(), 1U, 4U));
        const auto status = decoder->Init(&settings);
        if (status == libgav1::kStatusOk) {
            av1 = std::move(decoder);
        }
        LOG_INFO(HW_GPU, "nx2 video: libgav1 decoder for \"{}\" {} ({} threads)", name,
                 status == libgav1::kStatusOk ? "open" : libgav1::GetErrorString(status),
                 settings.threads);
        return status == libgav1::kStatusOk;
    }

    // A hardware decoder that takes 30 frames without putting out a picture, or puts out a layout
    // this can't read, is given up on for libgav1 (from the movie's next key frame).
    void DecodeHardwareAv1(std::span<const u8> data, u64 pts, std::vector<Nx2Picture>& pictures) {
        if (hardware_av1->Decode(data, pts, pictures) &&
            (hardware_av1->outputs != 0 || hardware_av1->inputs < 30)) {
            return;
        }
        LOG_WARNING(HW_GPU, "nx2 video: hardware AV1 decoder gave {} pictures for {} frames; "
                    "using libgav1", hardware_av1->outputs, hardware_av1->inputs);
        hardware_av1.reset();
        open = OpenLibgav1();
    }
#endif
    SwsContext* scaler{};
    bool open{};
    H264PacketConverter h264_packets;
    /// Timestamps of submitted frames not yet output. Frames copied back from a host GPU decoder
    /// lose theirs and take the earliest one instead.
    std::multiset<u64> pending_pts;
    std::vector<u8> packet; ///< FFmpeg reads up to AV_INPUT_BUFFER_PADDING_SIZE past the data

    ~Impl() {
        sws_freeContext(scaler);
        context.reset();
        hardware.reset();
    }

    void Receive(std::vector<Nx2Picture>& pictures) {
        while (auto frame = context->ReceiveFrame()) {
            const AVFrame* f = frame->GetFrame();
            const s64 stamp = f->best_effort_timestamp != AV_NOPTS_VALUE ? f->best_effort_timestamp
                                                                         : f->pts;
            u64 pts{};
            if (stamp != AV_NOPTS_VALUE) {
                pts = static_cast<u64>(stamp);
                if (const auto it = pending_pts.find(pts); it != pending_pts.end()) {
                    pending_pts.erase(it);
                }
            } else if (!pending_pts.empty()) {
                pts = *pending_pts.begin();
                pending_pts.erase(pending_pts.begin());
            }
            pictures.push_back({static_cast<u32>(f->width), static_cast<u32>(f->height), pts,
                                std::move(frame)});
        }
    }

#ifdef EDEN_NX2_LIBGAV1
    // Not in frame parallel mode, libgav1 decodes one temporal unit per DequeueFrame and holds
    // no picture back, so each one is decoded right away (EnqueueFrame keeps a pointer to it).
    void DecodeAv1(std::span<const u8> data, u64 pts, std::vector<Nx2Picture>& pictures) {
        auto status =
            av1->EnqueueFrame(data.data(), data.size(), static_cast<int64_t>(pts), nullptr);
        const libgav1::DecoderBuffer* buffer{};
        if (status == libgav1::kStatusOk) {
            status = av1->DequeueFrame(&buffer);
        }
        if (status != libgav1::kStatusOk) {
            if (av1_errors++ < 8 || (av1_errors % 600) == 0) {
                LOG_ERROR(HW_GPU, "nx2 video: AV1 frame of {} bytes not decoded ({}), {} so far",
                          data.size(), libgav1::GetErrorString(status), av1_errors);
            }
            return;
        }
        if (buffer == nullptr) {
            return; // nothing shown by this temporal unit
        }
        auto frame = CopyAv1Picture(*buffer);
        if (!frame) {
            if (av1_errors++ < 8) {
                LOG_ERROR(HW_GPU, "nx2 video: unsupported AV1 picture: {}-bit, format {}, {}x{}",
                          buffer->bitdepth, static_cast<int>(buffer->image_format),
                          buffer->displayed_width[0], buffer->displayed_height[0]);
            }
            return;
        }
        pictures.push_back({static_cast<u32>(buffer->displayed_width[0]),
                            static_cast<u32>(buffer->displayed_height[0]),
                            static_cast<u64>(buffer->user_private_data), std::move(frame)});
    }
#endif
};

Nx2VideoDecoder::Nx2VideoDecoder(std::string_view mime) : impl{std::make_unique<Impl>()} {
    impl->name = std::string{mime};
#ifdef EDEN_NX2_LIBGAV1
    // AV1 goes to libgav1 where FFmpeg has no libdav1d (Android's FFmpeg package); FFmpeg's own
    // "av1" decoder only drives hardware decoders.
    if (Nx2IsAv1Mime(mime) && avcodec_find_decoder_by_name("libdav1d") == nullptr) {
        impl->hardware_av1 = MediaCodecAv1::Create();
        if (impl->hardware_av1) {
            impl->open = true;
            LOG_INFO(HW_GPU, "nx2 video: hardware decoder {} for \"{}\"",
                     impl->hardware_av1->Name(), mime);
            return;
        }
        impl->open = impl->OpenLibgav1();
        return;
    }
#endif
    const auto codec = CodecFromMime(mime);
    if (!codec) {
        LOG_ERROR(HW_GPU, "nx2 video: no decoder for codec \"{}\"", mime);
        return;
    }
    impl->decoder.emplace(*codec);
    if (impl->decoder->GetCodec() == nullptr) {
        LOG_ERROR(HW_GPU, "nx2 video: FFmpeg has no decoder for \"{}\"", mime);
        return;
    }
    impl->context.emplace(*impl->decoder);
    if (Settings::values.nvdec_emulation.GetValue() == Settings::NvdecEmulation::Gpu) {
        impl->hardware.emplace();
        impl->hardware->InitializeForDecoder(*impl->context, *impl->decoder);
    }
    impl->open = impl->context->OpenContext(*impl->decoder);
    LOG_INFO(HW_GPU, "nx2 video: decoder for \"{}\" {}", mime, impl->open ? "open" : "failed to open");
}

Nx2VideoDecoder::~Nx2VideoDecoder() = default;

bool Nx2VideoDecoder::IsOpen() const {
    return impl->open;
}

const std::string& Nx2VideoDecoder::Name() const {
    return impl->name;
}

void Nx2VideoDecoder::Decode(std::span<const u8> data, u64 pts, std::vector<Nx2Picture>& pictures) {
    if (!impl->open || data.empty()) {
        return;
    }
#ifdef EDEN_NX2_LIBGAV1
    if (impl->hardware_av1) {
        impl->DecodeHardwareAv1(data, pts, pictures);
        return;
    }
    if (impl->av1) {
        impl->DecodeAv1(data, pts, pictures);
        return;
    }
#endif
    impl->packet.assign(data.size() + AV_INPUT_BUFFER_PADDING_SIZE, 0);
    std::memcpy(impl->packet.data(), data.data(), data.size());

    // Convert MP4/avcC length-prefixed NALUs to Annex-B start codes for FFmpeg if needed.
    if (impl->name.find("avc") != std::string::npos || impl->name.find("264") != std::string::npos) {
        impl->h264_packets.Convert(std::span<u8>{impl->packet.data(), data.size()});
    }

    FFmpeg::Packet packet{std::span<const u8>{impl->packet.data(), data.size()}};
    packet.GetPacket()->pts = static_cast<s64>(pts);
    // Hidden VP9 frames never come out: keep the set from growing over a long movie.
    if (impl->pending_pts.size() >= 64) {
        impl->pending_pts.erase(impl->pending_pts.begin());
    }
    impl->pending_pts.insert(pts);
    impl->context->SendPacket(packet);
    impl->Receive(pictures);
}

void Nx2VideoDecoder::Drain(std::vector<Nx2Picture>& pictures) {
    if (!impl->open) {
        return;
    }
#ifdef EDEN_NX2_LIBGAV1
    if (impl->hardware_av1) {
        impl->hardware_av1->Drain(pictures);
        return;
    }
    if (impl->av1) {
        Reset(); // libgav1 holds no picture back (see DecodeAv1)
        return;
    }
#endif
    avcodec_send_packet(impl->context->GetCodecContext(), nullptr);
    impl->Receive(pictures);
    Reset();
}

void Nx2VideoDecoder::Reset() {
#ifdef EDEN_NX2_LIBGAV1
    if (impl->hardware_av1) {
        impl->hardware_av1->Flush();
        impl->pending_pts.clear();
        return;
    }
    if (impl->av1) {
        // Releases its frames; the next frame may start a new sequence.
        impl->av1->SignalEOS();
        impl->pending_pts.clear();
        return;
    }
#endif
    if (impl->open) {
        avcodec_flush_buffers(impl->context->GetCodecContext());
    }
    impl->pending_pts.clear();
    impl->h264_packets.Reset();
}

void Nx2VideoDecoder::Render(const Nx2Picture& picture, Nx2PictureFormat format, u32 width,
                             u32 height, u64 pitch, u64 chroma_offset, std::span<u8> out) {
    const AVFrame* f = picture.frame->GetFrame();
    const u64 needed = format == Nx2PictureFormat::Nv12
                           ? chroma_offset + pitch * ((u64{height} + 1) / 2)
                           : pitch * height;
    const u64 min_pitch = format == Nx2PictureFormat::Nv12 ? ((u64{width} + 1) / 2) * 2 : 0;
    if (width == 0 || height == 0 || pitch < min_pitch || out.size() < needed) {
        return;
    }
    const AVPixelFormat target = format == Nx2PictureFormat::Nv12    ? AV_PIX_FMT_NV12
                                 : format == Nx2PictureFormat::Rgba8 ? AV_PIX_FMT_RGBA
                                                                     : AV_PIX_FMT_BGRA;
    // Frames copied back from a host GPU decoder are already NV12. Copy them as they are:
    // FFmpeg 6.1's swscale leaves the chroma plane of an NV12 to NV12 copy at zero, which shows
    // the movies green.
    if (target == AV_PIX_FMT_NV12 && f->format == AV_PIX_FMT_NV12 &&
        static_cast<u32>(f->width) == width && static_cast<u32>(f->height) == height) {
        for (u32 y = 0; y < height; ++y) {
            std::memcpy(out.data() + y * pitch, f->data[0] + y * f->linesize[0], width);
        }
        for (u32 y = 0; y < (height + 1) / 2; ++y) {
            std::memcpy(out.data() + chroma_offset + y * pitch, f->data[1] + y * f->linesize[1],
                        min_pitch);
        }
        return;
    }
    if (target == AV_PIX_FMT_NV12 && f->format == AV_PIX_FMT_YUV420P &&
        f->color_range != AVCOL_RANGE_JPEG && static_cast<u32>(f->width) == width &&
        static_cast<u32>(f->height) == height) {
        for (u32 y = 0; y < height; ++y) {
            std::memcpy(out.data() + y * pitch, f->data[0] + y * f->linesize[0], width);
        }
        const u32 chroma_width = static_cast<u32>(min_pitch / 2);
        for (u32 y = 0; y < (height + 1) / 2; ++y) {
            const u8* u = f->data[1] + y * f->linesize[1];
            const u8* v = f->data[2] + y * f->linesize[2];
            u8* dst = out.data() + chroma_offset + y * pitch;
            for (u32 x = 0; x < chroma_width; ++x) {
                dst[x * 2] = u[x];
                dst[x * 2 + 1] = v[x];
            }
        }
        return;
    }
    impl->scaler = sws_getCachedContext(impl->scaler, f->width, f->height,
                                        static_cast<AVPixelFormat>(f->format),
                                        static_cast<int>(width), static_cast<int>(height), target,
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (impl->scaler == nullptr) {
        LOG_ERROR(HW_GPU, "nx2 video: cannot convert {} to {}",
                  av_get_pix_fmt_name(static_cast<AVPixelFormat>(f->format)),
                  av_get_pix_fmt_name(target));
        return;
    }
    // The stream's matrix and range; untagged HD video is BT.709.
    int space = SWS_CS_ITU601;
    if (f->colorspace == AVCOL_SPC_BT709 ||
        (f->colorspace == AVCOL_SPC_UNSPECIFIED && f->height >= 720)) {
        space = SWS_CS_ITU709;
    } else if (f->colorspace == AVCOL_SPC_BT2020_NCL || f->colorspace == AVCOL_SPC_BT2020_CL) {
        space = SWS_CS_BT2020;
    }
    const bool full_range_source = f->color_range == AVCOL_RANGE_JPEG ||
                                  f->format == AV_PIX_FMT_YUVJ420P ||
                                  f->format == AV_PIX_FMT_YUVJ422P ||
                                  f->format == AV_PIX_FMT_YUVJ444P ||
                                  f->format == AV_PIX_FMT_YUVJ440P ||
                                  f->format == AV_PIX_FMT_YUVJ411P;
    const int* coefficients = sws_getCoefficients(space);
    sws_setColorspaceDetails(impl->scaler, coefficients, full_range_source, coefficients,
                             format != Nx2PictureFormat::Nv12, 0, 1 << 16, 1 << 16);
    std::array<u8*, 4> planes{out.data(), nullptr, nullptr, nullptr};
    std::array<int, 4> strides{static_cast<int>(pitch), 0, 0, 0};
    if (format == Nx2PictureFormat::Nv12) {
        planes[1] = out.data() + chroma_offset;
        strides[1] = static_cast<int>(pitch);
    }
    sws_scale(impl->scaler, f->data, f->linesize, 0, f->height, planes.data(), strides.data());
}

} // namespace Tegra::Host1x
