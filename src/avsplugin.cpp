/* 
 * Copyright © 2022 Frechdachs <frechdachs@rekt.cc>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <system_error>

extern "C" {
#include "descale.h"
#include "plugin.h"
}

#include "avs_c_api_loader.hpp"

struct AVSDescaleData
{
    std::once_flag init_flag;
    avs_helpers::avs_clip_ptr ignore_mask_clip;
    std::unique_ptr<double[]> post_conv;

    DescaleData dd{};

    ~AVSDescaleData()
    {
        if (dd.dsapi.free_core) {
            if (dd.process_h) {
                if (dd.dscore_h[0])
                    dd.dsapi.free_core(dd.dscore_h[0]);
                if (dd.dscore_h[1])
                    dd.dsapi.free_core(dd.dscore_h[1]);
            }
            if (dd.process_v) {
                if (dd.dscore_v[0])
                    dd.dsapi.free_core(dd.dscore_v[0]);
                if (dd.dscore_v[1])
                    dd.dsapi.free_core(dd.dscore_v[1]);
            }
        }
    }

    AVSDescaleData(const AVSDescaleData&) = delete;
    AVSDescaleData& operator=(const AVSDescaleData&) = delete;
    AVSDescaleData(AVSDescaleData&&) = delete;
    AVSDescaleData& operator=(AVSDescaleData&&) = delete;

    AVSDescaleData() = default;

    void ensure_initialized()
    {
        std::call_once(init_flag, [this]() { initialize_descale_data(&dd); });
    }
};

static AVS_VideoFrame* AVSC_CC avs_descale_get_frame(AVS_FilterInfo* fi, int n)
{
    auto* d = static_cast<AVSDescaleData*>(fi->user_data);
    d->ensure_initialized();

    constexpr std::array<int, 3> planes_rgb = {AVS_PLANAR_R, AVS_PLANAR_G, AVS_PLANAR_B};
    constexpr std::array<int, 3> planes_yuv = {AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V};
    const auto& planes = g_avs_api->avs_is_planar_rgb(&fi->vi) ? planes_rgb : planes_yuv;

    avs_helpers::avs_video_frame_ptr src(g_avs_api->avs_get_frame(fi->child, n));
    if (!src) {
        return nullptr;
    }
    avs_helpers::avs_video_frame_ptr ignore_mask;
    if (d->ignore_mask_clip) {
        ignore_mask.reset(g_avs_api->avs_get_frame(d->ignore_mask_clip.get(), n));
    }
    avs_helpers::avs_video_frame_ptr dst(g_avs_api->avs_new_video_frame_p_a(fi->env, &fi->vi, src.get(), 32));

    for (int i = 0; i < d->dd.num_planes; ++i) {
        const int plane = planes[i];
        const int dst_stride_bytes = g_avs_api->avs_get_pitch_p(dst.get(), plane);
        const int src_stride_pixels = g_avs_api->avs_get_pitch_p(src.get(), plane) / sizeof(float);
        const int dst_stride_pixels = dst_stride_bytes / sizeof(float);

        const auto* srcp = reinterpret_cast<const float*>(g_avs_api->avs_get_read_ptr_p(src.get(), plane));
        auto* dstp = reinterpret_cast<float*>(g_avs_api->avs_get_write_ptr_p(dst.get(), plane));

        int imask_stride_bytes = (ignore_mask) ? (g_avs_api->avs_get_pitch_p(ignore_mask.get(), plane) / sizeof(float)) : 0;
        const uint8_t* imaskp = (ignore_mask) ? (g_avs_api->avs_get_read_ptr_p(ignore_mask.get(), plane)) : nullptr;

        const bool is_chroma = (i > 0);
        const int subsample_h = is_chroma ? d->dd.subsampling_h : 0;
        const int subsample_v = is_chroma ? d->dd.subsampling_v : 0;
        const int core_idx_h = (is_chroma && d->dd.subsampling_h > 0) ? 1 : 0;
        const int core_idx_v = (is_chroma && d->dd.subsampling_v > 0) ? 1 : 0;

        if (d->dd.process_h && d->dd.process_v) {
            const int intermediate_height = d->dd.src_height >> subsample_v;
            const size_t intermediate_size = static_cast<size_t>(dst_stride_bytes) * intermediate_height;

            avs_helpers::avs_pool_ptr intermediatep(
                reinterpret_cast<std::byte*>(g_avs_api->avs_pool_allocate(fi->env, intermediate_size, 32, AVS_ALLOCTYPE_POOLED_ALLOC)),
                avs_helpers::avs_pool_deleter{fi->env}
            );

            auto* intermediate_float_p = reinterpret_cast<float*>(intermediatep.get());

            d->dd.dsapi.process_vectors(
                d->dd.dscore_h[core_idx_h], DESCALE_DIR_HORIZONTAL, intermediate_height, src_stride_pixels, 0, dst_stride_pixels, srcp,
                nullptr, intermediate_float_p
            );

            d->dd.dsapi.process_vectors(
                d->dd.dscore_v[core_idx_v], DESCALE_DIR_VERTICAL, d->dd.dst_width >> subsample_h, dst_stride_pixels, 0, dst_stride_pixels,
                intermediate_float_p, nullptr, dstp
            );
        } else if (d->dd.process_h) {
            d->dd.dsapi.process_vectors(
                d->dd.dscore_h[core_idx_h], DESCALE_DIR_HORIZONTAL, d->dd.src_height >> subsample_v, src_stride_pixels,
                (ignore_mask) ? (imask_stride_bytes) : 0, dst_stride_pixels, srcp, imaskp, dstp
            );

        } else if (d->dd.process_v) {
            d->dd.dsapi.process_vectors(
                d->dd.dscore_v[core_idx_v], DESCALE_DIR_VERTICAL, d->dd.src_width >> subsample_h, src_stride_pixels,
                (ignore_mask) ? (imask_stride_bytes) : 0, dst_stride_pixels, srcp, imaskp, dstp
            );
        }
    }

    return dst.release();
}

static int AVSC_CC avs_descale_set_cache_hints(AVS_FilterInfo* fi, int cachehints, int frame_range)
{
    return cachehints == AVS_CACHE_GET_MTMODE ? 1 /* MT_NICE_FILTER */ : 0;
}

static void AVSC_CC avs_descale_free(AVS_FilterInfo* fi)
{
    delete static_cast<AVSDescaleData*>(fi->user_data);
    fi->user_data = nullptr;
}

enum class FilterType
{
    GENERAL,
    BICUBIC,
    LANCZOS,
    FULL
};

struct ArgumentIndices
{
    int clip;
    int width;
    int height;
    int kernel;
    int taps;
    int b;
    int c;
    int blur;
    int post_conv;
    int src_left;
    int src_top;
    int src_width;
    int src_height;
    int border_handling;
    int ignore_mask;
    int opt;
};

constexpr std::array<ArgumentIndices, 4> layout_configs = {
    {{// GENERAL
      .clip = 0,
      .width = 1,
      .height = 2,
      .kernel = -1,
      .taps = -1,
      .b = -1,
      .c = -1,
      .blur = 3,
      .post_conv = 4,
      .src_left = 5,
      .src_top = 6,
      .src_width = 7,
      .src_height = 8,
      .border_handling = 9,
      .ignore_mask = 10,
      .opt = 11
     },
     {// BICUBIC
      .clip = 0,
      .width = 1,
      .height = 2,
      .kernel = -1,
      .taps = -1,
      .b = 3,
      .c = 4,
      .blur = 5,
      .post_conv = 6,
      .src_left = 7,
      .src_top = 8,
      .src_width = 9,
      .src_height = 10,
      .border_handling = 11,
      .ignore_mask = 12,
      .opt = 13
     },
     {// LANCZOS
      .clip = 0,
      .width = 1,
      .height = 2,
      .kernel = -1,
      .taps = 3,
      .b = -1,
      .c = -1,
      .blur = 4,
      .post_conv = 5,
      .src_left = 6,
      .src_top = 7,
      .src_width = 8,
      .src_height = 9,
      .border_handling = 10,
      .ignore_mask = 11,
      .opt = 12
     },
     {// FULL
      .clip = 0,
      .width = 1,
      .height = 2,
      .kernel = 3,
      .taps = 4,
      .b = 5,
      .c = 6,
      .blur = 7,
      .post_conv = 8,
      .src_left = 9,
      .src_top = 10,
      .src_width = 11,
      .src_height = 12,
      .border_handling = 13,
      .ignore_mask = 14,
      .opt = 15
     }}
};

static AVS_Value AVSC_CC avs_descale_create(AVS_ScriptEnvironment* env, AVS_Value args, void* user_data)
{
    DescaleMode mode = DESCALE_MODE_BILINEAR;

    const FilterType type = [&]() {
        if (user_data == nullptr) {
            return FilterType::FULL;
        } else {
            mode = static_cast<DescaleMode>(reinterpret_cast<intptr_t>(user_data));
            switch (mode) {
            case DESCALE_MODE_BILINEAR:
            case DESCALE_MODE_SPLINE16:
            case DESCALE_MODE_SPLINE36:
            case DESCALE_MODE_SPLINE64:
            case DESCALE_MODE_POINT:
                return FilterType::GENERAL;
            case DESCALE_MODE_BICUBIC:
                return FilterType::BICUBIC;
            case DESCALE_MODE_LANCZOS:
                return FilterType::LANCZOS;
            default:
                throw std::logic_error("this shouldn't be reached ever.");
            }
        }
    }();

    const ArgumentIndices& indices = layout_configs[static_cast<size_t>(type)];

    AVS_FilterInfo* fi{};
    avs_helpers::avs_clip_ptr clip_ref(g_avs_api->avs_new_c_filter(env, &fi, avs_array_elt(args, indices.clip), 1));

    if (!avs_has_video(&fi->vi)) {
        return avs_new_value_error("Descale: Input clip must have video.");
    }
    if (!g_avs_api->avs_is_y(&fi->vi) && !avs_is_yuv(&fi->vi) && !g_avs_api->avs_is_planar_rgb(&fi->vi)) {
        return avs_new_value_error("Descale: Input clip must be Y, YUV, or planar RGB.");
    }
    if (g_avs_api->avs_component_size(&fi->vi) < 4) {
        return avs_new_value_error("Descale: Input clip must be 32-bit float.");
    }

    std::unique_ptr<AVSDescaleData> data = std::make_unique<AVSDescaleData>();

    const int num_planes = g_avs_api->avs_num_components(&fi->vi);
    const int subsampling_h =
        (avs_is_yuv(&fi->vi) && !g_avs_api->avs_is_y(&fi->vi)) ? g_avs_api->avs_get_plane_width_subsampling(&fi->vi, AVS_PLANAR_U) : 0;
    const int subsampling_v =
        (avs_is_yuv(&fi->vi) && !g_avs_api->avs_is_y(&fi->vi)) ? g_avs_api->avs_get_plane_height_subsampling(&fi->vi, AVS_PLANAR_U) : 0;
    const int src_width = fi->vi.width;
    const int src_height = fi->vi.height;

    const int dst_width = avs_as_int(avs_array_elt(args, indices.width));
    const int dst_height = avs_as_int(avs_array_elt(args, indices.height));

    if (dst_width < 1) {
        return avs_new_value_error("Descale: width must be greater than 0.");
    }
    if (dst_height < 8) {
        return avs_new_value_error("Descale: Output height must be greater than or equal to 8.");
    }
    if ((dst_width % (1 << subsampling_h)) != 0) {
        return avs_new_value_error("Descale: Output width and output subsampling are not compatible.");
    }
    if ((dst_height % (1 << subsampling_v)) != 0) {
        return avs_new_value_error("Descale: Output height and output subsampling are not compatible.");
    }
    if (dst_width > src_width || dst_height > src_height) {
        return avs_new_value_error("Descale: Output dimension must be less than or equal to input dimension.");
    }

    if (type == FilterType::FULL) {
        std::string_view kernel_sv = avs_helpers::get_opt_arg<std::string_view>(env, args, indices.kernel).value();
        if (kernel_sv.empty())
            return avs_new_value_error("Descale: kernel argument is null.");

        auto string_view_equals_ignore_case = [&](std::string_view a, std::string_view b) {
            return std::ranges::equal(a, b, [](char ca, char cb) {
                return std::tolower(static_cast<unsigned char>(ca)) == std::tolower(static_cast<unsigned char>(cb));
            });
        };

        if (string_view_equals_ignore_case(kernel_sv, "bilinear"))
            mode = DESCALE_MODE_BILINEAR;
        else if (string_view_equals_ignore_case(kernel_sv, "bicubic"))
            mode = DESCALE_MODE_BICUBIC;
        else if (string_view_equals_ignore_case(kernel_sv, "lanczos"))
            mode = DESCALE_MODE_LANCZOS;
        else if (string_view_equals_ignore_case(kernel_sv, "spline16"))
            mode = DESCALE_MODE_SPLINE16;
        else if (string_view_equals_ignore_case(kernel_sv, "spline36"))
            mode = DESCALE_MODE_SPLINE36;
        else if (string_view_equals_ignore_case(kernel_sv, "spline64"))
            mode = DESCALE_MODE_SPLINE64;
        else if (string_view_equals_ignore_case(kernel_sv, "point"))
            mode = DESCALE_MODE_POINT;
        else
            return avs_new_value_error("Descale: Invalid kernel specified.");
    }

    const int taps = (indices.taps >= 0) ? avs_helpers::get_opt_arg<int>(env, args, indices.taps).value_or(3) : 3;
    if (mode == DESCALE_MODE_LANCZOS && taps < 1) {
        return avs_new_value_error("Descale: taps must be greater than 0 for Lanczos.");
    }

    const double b = (indices.b >= 0) ? avs_helpers::get_opt_arg<double>(env, args, indices.b).value_or(0.0) : 0.0;
    const double c = (indices.c >= 0) ? avs_helpers::get_opt_arg<double>(env, args, indices.c).value_or(0.5) : 0.5;

    bool has_ignore_mask = false;
    std::optional<avs_helpers::avs_clip_ptr> mask_clip_raw =
        avs_helpers::get_opt_arg<avs_helpers::avs_clip_ptr>(env, args, indices.ignore_mask);
    if (mask_clip_raw) {
        data->ignore_mask_clip = std::move(*mask_clip_raw);
        const AVS_VideoInfo* mvi = g_avs_api->avs_get_video_info(data->ignore_mask_clip.get());
        if (!((fi->vi.pixel_type == mvi->pixel_type) || (g_avs_api->avs_is_yv12(&fi->vi) && g_avs_api->avs_is_yv12(mvi)))) {
            return avs_new_value_error("Descale: Ignore mask format must match clip format.");
        }
        if (fi->vi.width != mvi->width || fi->vi.height != mvi->height) {
            return avs_new_value_error("Descale: Ignore mask dimensions must match clip dimensions.");
        }
        if (fi->vi.num_frames != mvi->num_frames) {
            return avs_new_value_error("Descale: Ignore mask frame count must match clip frame count.");
        }
        if (g_avs_api->avs_component_size(mvi) < 4) {
            return avs_new_value_error("Descale: Ignore mask clip must be 32-bit float.");
        }
        has_ignore_mask = true;
    }

    const double shift_h = avs_helpers::get_opt_arg<double>(env, args, indices.src_left).value_or(0.0);
    const double shift_v = avs_helpers::get_opt_arg<double>(env, args, indices.src_top).value_or(0.0);
    const double active_width = avs_helpers::get_opt_arg<double>(env, args, indices.src_width).value_or(static_cast<double>(dst_width));
    const double active_height = avs_helpers::get_opt_arg<double>(env, args, indices.src_height).value_or(static_cast<double>(dst_height));

    const double blur = avs_helpers::get_opt_arg<double>(env, args, indices.blur).value_or(1.0);
    if (blur >= src_width >> subsampling_h || blur >= src_height >> subsampling_v || blur <= 0.0) {
        // We also need to ensure that the blur isn't smaller than 1 / support, but we can't know the exact support of the kernel here,
        return avs_new_value_error("Descale: blur parameter is out of bounds.");
    }

    const bool process_h = dst_width != src_width || shift_h != 0.0 || active_width != static_cast<double>(dst_width);
    const bool process_v = dst_height != src_height || shift_v != 0.0 || active_height != static_cast<double>(dst_height);

    if (!process_h && !process_v) {
        AVS_Value v;
        g_avs_api->avs_set_to_clip(&v, clip_ref.get());
        return v;
    }

    if (process_h && process_v && data->ignore_mask_clip) {
        return avs_new_value_error("Descale: Ignore mask is not supported when descaling along both axes.");
    }

    const int border_handling = avs_helpers::get_opt_arg<int>(env, args, indices.border_handling).value_or(0);
    enum DescaleBorder border_handling_enum = [&]() {
        switch (border_handling) {
        case 1:
            return DESCALE_BORDER_ZERO;
        case 2:
            return DESCALE_BORDER_REPEAT;
        default:
            return DESCALE_BORDER_MIRROR;
        }
    }();

    const int opt = avs_helpers::get_opt_arg<int>(env, args, indices.opt).value_or(0);
    enum DescaleOpt opt_enum = [&]() {
        if (data->ignore_mask_clip)
            return DESCALE_OPT_NONE;
        switch (opt) {
        case 1:
            return DESCALE_OPT_NONE;
        case 2:
            return DESCALE_OPT_AVX2;
        default:
            return DESCALE_OPT_AUTO;
        }
    }();

    avs_helpers::converted_array<double> post_conv_load{avs_helpers::get_opt_array_as_unique_ptr<double>(args, indices.post_conv)};
    const int post_conv_size{post_conv_load.size};

    if (post_conv_size > 0) {
        if (post_conv_size % 2 != 1) {
            return avs_new_value_error("Post-convolution kernel must have odd length.");
        }
        if ((process_h && post_conv_size > 2 * fi->vi.width + 1) || (process_v && post_conv_size > 2 * fi->vi.height + 1)) {
            return avs_new_value_error("Post-convolution kernel is too large, exceeds clip dimensions.");
        }

        data->post_conv = std::move(post_conv_load.data);
    }

    DescaleData& dd = data->dd;
    dd.params = DescaleParams{
        .mode = mode,
        .upscale = 0,
        .taps = taps,
        .param1 = b,
        .param2 = c,
        .blur = blur,
        .post_conv_size = post_conv_size,
        .post_conv = data->post_conv.get(),
        .shift = 0.0,
        .active_dim = 0.0,
        .has_ignore_mask = has_ignore_mask,
        .border_handling = border_handling_enum,
        .custom_kernel = nullptr
    };

    dd.src_width = src_width;
    dd.src_height = src_height;
    dd.dst_width = dst_width;
    dd.dst_height = dst_height;
    dd.subsampling_h = subsampling_h;
    dd.subsampling_v = subsampling_v;
    dd.num_planes = num_planes;
    dd.process_h = process_h;
    dd.process_v = process_v;
    dd.shift_h = shift_h;
    dd.shift_v = shift_v;
    dd.active_width = active_width;
    dd.active_height = active_height;
    dd.dsapi = get_descale_api(opt_enum);

    fi->vi.width = dst_width;
    fi->vi.height = dst_height;
    fi->user_data = data.release();
    fi->get_frame = avs_descale_get_frame;
    fi->set_cache_hints = avs_descale_set_cache_hints;
    fi->free_filter = avs_descale_free;

    AVS_Value v;
    g_avs_api->avs_set_to_clip(&v, clip_ref.get());

    return v;
}

const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env)
{
    static constexpr int REQUIRED_INTERFACE_VERSION{9};
    static constexpr int REQUIRED_BUGFIX_VERSION{2};
    static constexpr std::string_view required_functions_storage[]{
        "avs_pool_free",           // avs loader helper functions
        "avs_release_clip",        // avs loader helper functions
        "avs_release_value",       // avs loader helper functions
        "avs_release_video_frame", // avs loader helper functions
        "avs_take_clip",           // avs loader helper functions
        "avs_add_function",
        "avs_component_size",
        "avs_get_frame",
        "avs_get_pitch_p",
        "avs_get_plane_height_subsampling",
        "avs_get_plane_width_subsampling",
        "avs_get_read_ptr_p",
        "avs_get_video_info",
        "avs_get_write_ptr_p",
        "avs_is_planar_rgb",
        "avs_is_y",
        "avs_is_yv12",
        "avs_new_c_filter",
        "avs_new_video_frame_p_a",
        "avs_num_components",
        "avs_pool_allocate",
        "avs_set_to_clip"
    };
    static constexpr std::span<const std::string_view> required_functions{ required_functions_storage };

    if (!avisynth_c_api_loader::get_api(env, REQUIRED_INTERFACE_VERSION, REQUIRED_BUGFIX_VERSION, required_functions)) {
        std::cerr << avisynth_c_api_loader::get_last_error() << std::endl;
        return avisynth_c_api_loader::get_last_error();
    }

    auto add_func = [&](const char* name, const char* params, DescaleMode mode) {
        g_avs_api->avs_add_function(env, name, params, avs_descale_create, reinterpret_cast<void*>(static_cast<intptr_t>(mode)));
    };

    constexpr const char* general_sig = "c"
                                        "i"
                                        "i"
                                        "[blur]f"
                                        "[post_conv]f*"
                                        "[src_left]f"
                                        "[src_top]f"
                                        "[src_width]f"
                                        "[src_height]f"
                                        "[border_handling]i"
                                        "[ignore_mask]c"
                                        "[opt]i";
    constexpr const char* bicubic_sig = "c"
                                        "i"
                                        "i"
                                        "[b]f"
                                        "[c]f"
                                        "[blur]f"
                                        "[post_conv]f*"
                                        "[src_left]f"
                                        "[src_top]f"
                                        "[src_width]f"
                                        "[src_height]f"
                                        "[border_handling]i"
                                        "[ignore_mask]c"
                                        "[opt]i";
    constexpr const char* lanczos_sig = "c"
                                        "i"
                                        "i"
                                        "[taps]i"
                                        "[blur]f"
                                        "[post_conv]f*"
                                        "[src_left]f"
                                        "[src_top]f"
                                        "[src_width]f"
                                        "[src_height]f"
                                        "[border_handling]i"
                                        "[ignore_mask]c"
                                        "[opt]i";
    constexpr const char* full_sig = "c"
                                     "i"
                                     "i"
                                     "[kernel]s"
                                     "[taps]i"
                                     "[b]f"
                                     "[c]f"
                                     "[blur]f"
                                     "[post_conv]f*"
                                     "[src_left]f"
                                     "[src_top]f"
                                     "[src_width]f"
                                     "[src_height]f"
                                     "[border_handling]i"
                                     "[ignore_mask]c"
                                     "[opt]i";

    add_func("Debilinear", general_sig, DESCALE_MODE_BILINEAR);
    add_func("Debicubic", bicubic_sig, DESCALE_MODE_BICUBIC);
    add_func("Delanczos", lanczos_sig, DESCALE_MODE_LANCZOS);
    add_func("Despline16", general_sig, DESCALE_MODE_SPLINE16);
    add_func("Despline36", general_sig, DESCALE_MODE_SPLINE36);
    add_func("Despline64", general_sig, DESCALE_MODE_SPLINE64);
    add_func("Depoint", general_sig, DESCALE_MODE_POINT);

    g_avs_api->avs_add_function(env, "Descale", full_sig, avs_descale_create, nullptr);

    return "Descale plugin";
}
