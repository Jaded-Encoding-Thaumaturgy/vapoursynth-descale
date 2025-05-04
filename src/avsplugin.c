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


#ifndef _MSC_VER
#include <pthread.h>
#endif // !_MSC_VER

#include <stdbool.h>
#include <stdlib.h>
#include <avisynth_c.h>
#include "descale.h"
#include "plugin.h"

#ifdef _MSC_VER
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif // !WIN32_LEAN_AND_MEAN

#include <windows.h>

typedef CRITICAL_SECTION pthread_mutex_t;

static inline int portable_mutex_init(pthread_mutex_t* mutex) {
    InitializeCriticalSection(mutex);
    return 0;
}

static inline int pthread_mutex_destroy(pthread_mutex_t* mutex) {
    DeleteCriticalSection(mutex);
    return 0;
}

static inline int pthread_mutex_lock(pthread_mutex_t* mutex) {
    EnterCriticalSection(mutex);
    return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t* mutex) {
    LeaveCriticalSection(mutex);
    return 0;
}
#else
static inline int portable_mutex_init(pthread_mutex_t* mutex) {
    return pthread_mutex_init(mutex, NULL);
}
#endif // _MSC_VER

struct AVSDescaleData
{
    bool initialized;
    pthread_mutex_t lock;
    AVS_Clip* ignore_mask;

    struct DescaleData dd;
};


static AVS_VideoFrame * AVSC_CC avs_descale_get_frame(AVS_FilterInfo *fi, int n)
{
    struct AVSDescaleData *d = (struct AVSDescaleData *)fi->user_data;

    if (!d->initialized) {
        pthread_mutex_lock(&d->lock);
        if (!d->initialized) {
            initialize_descale_data(&d->dd);
            d->initialized = true;
        }
        pthread_mutex_unlock(&d->lock);
    }

    // What the fuck is this shit?! Why not just index the planes with 0, 1, 2?
    int planes_rgb[] = {AVS_PLANAR_R, AVS_PLANAR_G, AVS_PLANAR_B};
    int planes_yuv[] = {AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V};
    int *planes;
    if (avs_is_planar_rgb(&fi->vi))
        planes = planes_rgb;
    else
        planes = planes_yuv;

    AVS_VideoFrame *src = avs_get_frame(fi->child, n);
    if (!src)
        return NULL;
    AVS_VideoFrame* ignore_mask = (d->ignore_mask) ? avs_get_frame(d->ignore_mask, n) : NULL;
    AVS_VideoFrame *dst = avs_new_video_frame_p_a(fi->env, &fi->vi, src, 32);

    for (int i = 0; i < d->dd.num_planes; i++) {
        int plane = planes[i];
        int src_stride = avs_get_pitch_p(src, plane) / sizeof(float);
        int dst_stride = avs_get_pitch_p(dst, plane) / sizeof(float);
        const float* srcp = (const float*)avs_get_read_ptr_p(src, plane);
        float* dstp = (float*)avs_get_write_ptr_p(dst, plane);
        const int imask_stride = (d->ignore_mask) ? (avs_get_pitch_p(ignore_mask, plane) / sizeof(float)) : 0;
        const uint8_t* imaskp = (d->ignore_mask) ? avs_get_read_ptr_p(ignore_mask, plane) : NULL;

        if (d->dd.process_h && d->dd.process_v) {
            int intermediate_stride = avs_get_pitch_p(dst, plane);
            float* intermediatep =
                avs_pool_allocate(fi->env, intermediate_stride * d->dd.src_height * sizeof(float), 32, AVS_ALLOCTYPE_POOLED_ALLOC);

            d->dd.dsapi.process_vectors(d->dd.dscore_h[i && d->dd.subsampling_h], DESCALE_DIR_HORIZONTAL,
                d->dd.src_height >> (i ? d->dd.subsampling_v : 0), src_stride, 0, intermediate_stride, srcp, NULL, intermediatep);
            d->dd.dsapi.process_vectors(d->dd.dscore_v[i && d->dd.subsampling_v], DESCALE_DIR_VERTICAL,
                d->dd.dst_width >> (i ? d->dd.subsampling_h : 0), intermediate_stride, 0, dst_stride, intermediatep, NULL, dstp);

            avs_pool_free(fi->env, intermediatep);

        } else if (d->dd.process_h) {
            d->dd.dsapi.process_vectors(d->dd.dscore_h[i && d->dd.subsampling_h], DESCALE_DIR_HORIZONTAL,
                d->dd.src_height >> (i ? d->dd.subsampling_v : 0), src_stride, imask_stride, dst_stride, srcp, imaskp, dstp);

        } else if (d->dd.process_v) {
            d->dd.dsapi.process_vectors(d->dd.dscore_v[i && d->dd.subsampling_v], DESCALE_DIR_VERTICAL,
                d->dd.src_width >> (i ? d->dd.subsampling_h : 0), src_stride, imask_stride, dst_stride, srcp, imaskp, dstp);
        }
    }

    if (ignore_mask)
        avs_release_video_frame(ignore_mask);
    avs_release_video_frame(src);

    return dst;
}


static int AVSC_CC avs_descale_set_cache_hints(AVS_FilterInfo *fi, int cachehints, int frame_range)
{
    return cachehints == AVS_CACHE_GET_MTMODE ? 1 /* MT_NICE_FILTER */ : 0;
}


static void AVSC_CC avs_descale_free(AVS_FilterInfo *fi)
{
    struct AVSDescaleData *d = (struct AVSDescaleData *)fi->user_data;

    if (d->dd.params.post_conv)
        free(d->dd.params.post_conv);
    if (d->ignore_mask)
        avs_release_clip(d->ignore_mask);
    if (d->initialized) {
        if (d->dd.process_h) {
            d->dd.dsapi.free_core(d->dd.dscore_h[0]);
            if (d->dd.num_planes > 1 && d->dd.subsampling_h > 0)
                d->dd.dsapi.free_core(d->dd.dscore_h[1]);
        }
        if (d->dd.process_v) {
            d->dd.dsapi.free_core(d->dd.dscore_v[0]);
            if (d->dd.num_planes > 1 && d->dd.subsampling_v > 0)
                d->dd.dsapi.free_core(d->dd.dscore_v[1]);
        }
    }

    pthread_mutex_destroy(&d->lock);

    free(d);
}

enum ArgsLayoutDescaleGeneral
{
    DescaleGeneral_clip,
    DescaleGeneral_width,
    DescaleGeneral_height,
    DescaleGeneral_blur,
    DescaleGeneral_post_conv,
    DescaleGeneral_src_left,
    DescaleGeneral_src_top,
    DescaleGeneral_src_width,
    DescaleGeneral_src_height,
    DescaleGeneral_border_handling,
    DescaleGeneral_ignore_mask,
    DescaleGeneral_opt
};

enum ArgsLayoutDescaleBicubic
{
    DescaleBicubic_clip,
    DescaleBicubic_width,
    DescaleBicubic_height,
    DescaleBicubic_b,
    DescaleBicubic_c,
    DescaleBicubic_blur,
    DescaleBicubic_post_conv,
    DescaleBicubic_src_left,
    DescaleBicubic_src_top,
    DescaleBicubic_src_width,
    DescaleBicubic_src_height,
    DescaleBicubic_border_handling,
    DescaleBicubic_ignore_mask,
    DescaleBicubic_opt
};

enum ArgsLayoutDescaleLanczos
{
    DescaleLanczos_clip,
    DescaleLanczos_width,
    DescaleLanczos_height,
    DescaleLanczos_taps,
    DescaleLanczos_blur,
    DescaleLanczos_post_conv,
    DescaleLanczos_src_left,
    DescaleLanczos_src_top,
    DescaleLanczos_src_width,
    DescaleLanczos_src_height,
    DescaleLanczos_border_handling,
    DescaleLanczos_ignore_mask,
    DescaleLanczos_opt
};

enum ArgsLayoutDescale
{
    Descale_clip,
    Descale_width,
    Descale_height,
    Descale_kernel,
    Descale_taps,
    Descale_b,
    Descale_c,
    Descale_blur,
    Descale_post_conv,
    Descale_src_left,
    Descale_src_top,
    Descale_src_width,
    Descale_src_height,
    Descale_border_handling,
    Descale_ignore_mask,
    Descale_opt
};

typedef struct
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
} ArgumentIndices;

typedef enum
{
    FILTER_TYPE_GENERAL,
    FILTER_TYPE_BICUBIC,
    FILTER_TYPE_LANCZOS,
    FILTER_TYPE_FULL
} FilterType;

static const ArgumentIndices layout_configs[4] = {
    [FILTER_TYPE_GENERAL] = {.clip = DescaleGeneral_clip,
        .width = DescaleGeneral_width,
        .height = DescaleGeneral_height,
        .kernel = -1,
        .taps = -1,
        .b = -1,
        .c = -1,
        .blur = DescaleGeneral_blur,
        .post_conv = DescaleGeneral_post_conv,
        .src_left = DescaleGeneral_src_left,
        .src_top = DescaleGeneral_src_top,
        .src_width = DescaleGeneral_src_width,
        .src_height = DescaleGeneral_src_height,
        .border_handling = DescaleGeneral_border_handling,
        .ignore_mask = DescaleGeneral_ignore_mask,
        .opt = DescaleGeneral_opt},
    [FILTER_TYPE_BICUBIC] = {.clip = DescaleBicubic_clip,
        .width = DescaleBicubic_width,
        .height = DescaleBicubic_height,
        .kernel = -1,
        .taps = -1,
        .b = DescaleBicubic_b,
        .c = DescaleBicubic_c,
        .blur = DescaleBicubic_blur,
        .post_conv = DescaleBicubic_post_conv,
        .src_left = DescaleBicubic_src_left,
        .src_top = DescaleBicubic_src_top,
        .src_width = DescaleBicubic_src_width,
        .src_height = DescaleBicubic_src_height,
        .border_handling = DescaleBicubic_border_handling,
        .ignore_mask = DescaleBicubic_ignore_mask,
        .opt = DescaleBicubic_opt},
    [FILTER_TYPE_LANCZOS] = {.clip = DescaleLanczos_clip,
        .width = DescaleLanczos_width,
        .height = DescaleLanczos_height,
        .kernel = -1,
        .taps = DescaleLanczos_taps,
        .b = -1,
        .c = -1,
        .blur = DescaleLanczos_blur,
        .post_conv = DescaleLanczos_post_conv,
        .src_left = DescaleLanczos_src_left,
        .src_top = DescaleLanczos_src_top,
        .src_width = DescaleLanczos_src_width,
        .src_height = DescaleLanczos_src_height,
        .border_handling = DescaleLanczos_border_handling,
        .ignore_mask = DescaleLanczos_ignore_mask,
        .opt = DescaleLanczos_opt},
    [FILTER_TYPE_FULL] = {.clip = Descale_clip,
        .width = Descale_width,
        .height = Descale_height,
        .kernel = Descale_kernel,
        .taps = Descale_taps,
        .b = Descale_b,
        .c = Descale_c,
        .blur = Descale_blur,
        .post_conv = Descale_post_conv,
        .src_left = Descale_src_left,
        .src_top = Descale_src_top,
        .src_width = Descale_src_width,
        .src_height = Descale_src_height,
        .border_handling = Descale_border_handling,
        .ignore_mask = Descale_ignore_mask,
        .opt = Descale_opt}};

static AVS_Value AVSC_CC avs_descale_create(AVS_ScriptEnvironment *env, AVS_Value args, void *user_data)
{
    FilterType type;
    enum DescaleMode mode;
    if (user_data == NULL)
        type = FILTER_TYPE_FULL;
    else {
        mode = (enum DescaleMode)user_data;
        if (mode == DESCALE_MODE_BILINEAR)
            type = FILTER_TYPE_GENERAL;
        else if (mode == DESCALE_MODE_BICUBIC)
            type = FILTER_TYPE_BICUBIC;
        else if (mode == DESCALE_MODE_LANCZOS)
            type = FILTER_TYPE_LANCZOS;
        else if (mode == DESCALE_MODE_SPLINE16)
            type = FILTER_TYPE_GENERAL;
        else if (mode == DESCALE_MODE_SPLINE36)
            type = FILTER_TYPE_GENERAL;
        else if (mode == DESCALE_MODE_SPLINE64)
            type = FILTER_TYPE_GENERAL;
        else if (mode == DESCALE_MODE_POINT)
            type = FILTER_TYPE_GENERAL;
    }

    const ArgumentIndices indices = layout_configs[type];
    AVS_FilterInfo* fi;
    struct AVSDescaleData* data = calloc(1, sizeof(struct AVSDescaleData));
    AVS_Clip* clip = avs_new_c_filter(env, &fi, avs_array_elt(args, indices.clip), 1);
    AVS_Value v = avs_void;

    if (!avs_has_video(&fi->vi)) {
        v = avs_new_value_error("Descale: Input clip must have video.");
        goto done;
    }

    if (!avs_is_y(&fi->vi) && !avs_is_yuv(&fi->vi) && !avs_is_planar_rgb(&fi->vi)) {
        v = avs_new_value_error("Descale: Input clip must be Y, YUV, or planar RGB.");
        goto done;
    }

    if (avs_component_size(&fi->vi) < 4) {
        v = avs_new_value_error("Descale: Input clip must be 32-bit float.");
        goto done;
    }

    int num_planes = avs_num_components(&fi->vi);

    // Apparently is_yuv() returns true for Y input.
    int subsampling_h = avs_is_yuv(&fi->vi) && !avs_is_y(&fi->vi) ? avs_get_plane_width_subsampling(&fi->vi, AVS_PLANAR_U) : 0;
    int subsampling_v = avs_is_yuv(&fi->vi) && !avs_is_y(&fi->vi) ? avs_get_plane_height_subsampling(&fi->vi, AVS_PLANAR_U) : 0;

    int src_width = fi->vi.width;
    int src_height = fi->vi.height;

    v = avs_array_elt(args, indices.width);
    int dst_width = avs_as_int(v);
    v = avs_array_elt(args, indices.height);
    int dst_height = avs_as_int(v);

    if (dst_width < 1) {
        v = avs_new_value_error("Descale: width must be greater than 0.");
        goto done;
    }
    if (dst_height < 8) {
        v = avs_new_value_error("Descale: Output height must be greater than or equal to 8.");
        goto done;
    }

    if (dst_width % (1 << subsampling_h) != 0) {
        v = avs_new_value_error("Descale: Output width and output subsampling are not compatible.");
        goto done;
    }
    if (dst_height % (1 << subsampling_v) != 0) {
        v = avs_new_value_error("Descale: Output height and output subsampling are not compatible.");
        goto done;
    }

    if (dst_width > src_width || dst_height > src_height) {
        v = avs_new_value_error("Descale: Output dimension must be less than or equal to input dimension.");
        goto done;
    }

    if (user_data == NULL) {
        v = avs_array_elt(args, indices.kernel);
        if (!avs_defined(v)) {
            v = avs_new_value_error("Descale: kernel is a required argument.");
            goto done;
        }
        const char* kernel = avs_as_string(v);
        if (string_is_equal_ignore_case(kernel, "bilinear"))
            mode = DESCALE_MODE_BILINEAR;
        else if (string_is_equal_ignore_case(kernel, "bicubic"))
            mode = DESCALE_MODE_BICUBIC;
        else if (string_is_equal_ignore_case(kernel, "lanczos"))
            mode = DESCALE_MODE_LANCZOS;
        else if (string_is_equal_ignore_case(kernel, "spline16"))
            mode = DESCALE_MODE_SPLINE16;
        else if (string_is_equal_ignore_case(kernel, "spline36"))
            mode = DESCALE_MODE_SPLINE36;
        else if (string_is_equal_ignore_case(kernel, "spline64"))
            mode = DESCALE_MODE_SPLINE64;
        else if (string_is_equal_ignore_case(kernel, "point"))
            mode = DESCALE_MODE_POINT;
        else {
            v = avs_new_value_error("Descale: Invalid kernel specified.");
            goto done;
        }
    }

    int taps = 3;
    if (user_data == NULL || mode == DESCALE_MODE_LANCZOS) {
        v = avs_array_elt(args, indices.taps);
        taps = avs_defined(v) ? avs_as_int(v) : 3;
        if (mode == DESCALE_MODE_LANCZOS && taps < 1) {
            v = avs_new_value_error("Descale: taps must be greater than 0.");
            goto done;
        }
    }

    double b = 0.0;
    double c = 0.5;
    if (user_data == NULL || mode == DESCALE_MODE_BICUBIC) {
        v = avs_array_elt(args, indices.b);
        b = avs_defined(v) ? avs_as_float(v) : 0.0;
        v = avs_array_elt(args, indices.c);
        c = avs_defined(v) ? avs_as_float(v) : 0.5;
    }

    int has_ignore_mask = 0;
    v = avs_array_elt(args, indices.ignore_mask);
    if (!avs_defined(v)) {
        data->ignore_mask = NULL;
    } else {
        has_ignore_mask = 1;
        const AVS_VideoInfo* mvi = avs_get_video_info(data->ignore_mask);
        if (!avs_is_same_colorspace(&fi->vi, mvi)) {
            v = avs_new_value_error("Descale: Ignore mask format must match clip format.");
            goto done;
        }
        if (fi->vi.width != mvi->width || fi->vi.height != mvi->height) {
            v = avs_new_value_error("Descale: Ignore mask dimensions must match clip dimensions.");
            goto done;
        }
        if (fi->vi.num_frames != mvi->num_frames) {
            v = avs_new_value_error("Descale: Ignore mask frames number must match clip frames number.");
            goto done;
        }
    }

    v = avs_array_elt(args, indices.src_left);
    double shift_h = avs_defined(v) ? avs_as_float(v) : 0.0;
    v = avs_array_elt(args, indices.src_top);
    double shift_v = avs_defined(v) ? avs_as_float(v) : 0.0;
    v = avs_array_elt(args, indices.src_width);
    double active_width = avs_defined(v) ? avs_as_float(v) : (double)dst_width;
    v = avs_array_elt(args, indices.src_height);
    double active_height = avs_defined(v) ? avs_as_float(v) : (double)dst_height;
    v = avs_array_elt(args, indices.blur);
    double blur = avs_defined(v) ? avs_as_float(v) : 1.0;
    if (blur >= src_width >> subsampling_h || blur >= src_height >> subsampling_v || blur <= 0.0) {
        // We also need to ensure that the blur isn't smaller than 1 / support, but we can't know the exact support of the kernel here,
        v = avs_new_value_error("Descale: blur parameter is out of bounds.");
        goto done;
    }

    bool process_h = dst_width != src_width || shift_h != 0.0 || active_width != (double)dst_width;
    bool process_v = dst_height != src_height || shift_v != 0.0 || active_height != (double)dst_height;

    if (!process_h && !process_v) {
        v = avs_new_value_clip(clip);
        goto done;
    }
    
    if (process_h && process_v && data->ignore_mask) {
        v = avs_new_value_error("Descale: Ignore mask is not supported when descaling along both axes.");
        goto done;
    }

    v = avs_array_elt(args, indices.border_handling);
    int border_handling = avs_defined(v) ? avs_as_int(v) : 0;
    enum DescaleBorder border_handling_enum;
    if (border_handling == 1)
        border_handling_enum = DESCALE_BORDER_ZERO;
    else if (border_handling == 2)
        border_handling_enum = DESCALE_BORDER_REPEAT;
    else
        border_handling_enum = DESCALE_BORDER_MIRROR;

    v = avs_array_elt(args, indices.opt);
    int opt = avs_defined(v) ? avs_as_int(v) : 0;
    enum DescaleOpt opt_enum;
    if (opt == 1)
        opt_enum = DESCALE_OPT_NONE;
    else if (opt == 2)
        opt_enum = DESCALE_OPT_AVX2;
    else
        opt_enum = DESCALE_OPT_AUTO;

    if (data->ignore_mask)
        opt_enum = DESCALE_OPT_NONE;
    
    double* post_conv = NULL;
    v = avs_array_elt(args, indices.post_conv);
    const int post_conv_size = (avs_defined(v)) ? avs_get_array_size(v) : 0;
    if (post_conv_size) {
        if (post_conv_size % 2 != 1) {
            v = avs_new_value_error("Post-convolution kernel must have odd length.");
            goto done;
        }
        if ((process_h && post_conv_size > 2 * fi->vi.width + 1) || (process_v && post_conv_size > 2 * fi->vi.height + 1)) {
            v = avs_new_value_error("Post-convolution kernel is too large, exceeds clip dimensions.");
            goto done;
        }

        post_conv = calloc(post_conv_size, sizeof(double));
        for (int i = 0; i < post_conv_size; i++) {
            post_conv[i] = avs_as_float(*(avs_as_array(v) + i));
        }
    }

    struct DescaleParams params = {.mode = mode,
        .upscale = 0,
        .taps = taps,
        .param1 = b,
        .param2 = c,
        .blur = blur,
        .post_conv_size = post_conv_size,
        .post_conv = post_conv,
        .shift = 0.0,
        .active_dim = 0.0,
        .has_ignore_mask = has_ignore_mask,
        .border_handling = border_handling_enum,
        .custom_kernel = NULL};
    struct DescaleData dd = {.src_width = src_width,
        .src_height = src_height,
        .dst_width = dst_width,
        .dst_height = dst_height,
        .subsampling_h = subsampling_h,
        .subsampling_v = subsampling_v,
        .num_planes = num_planes,
        .process_h = process_h,
        .process_v = process_v,
        .shift_h = shift_h,
        .shift_v = shift_v,
        .active_width = active_width,
        .active_height = active_height,
        .dsapi = get_descale_api(opt_enum),
        .params = params,
        {NULL, NULL},
        {NULL, NULL}};

    data->dd = dd;
    portable_mutex_init(&data->lock);

    fi->vi.width = dst_width;
    fi->vi.height = dst_height;
    fi->user_data = data;
    fi->get_frame = &avs_descale_get_frame;
    fi->set_cache_hints = &avs_descale_set_cache_hints;
    fi->free_filter = &avs_descale_free;

    v = avs_new_value_clip(clip);

done:
    avs_release_clip(clip);
    if (data->ignore_mask)
        avs_release_clip(data->ignore_mask);
    return v;
}


const char * AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment *env)
{
    avs_add_function(
        env,
        "Debilinear",
        "c"
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
        "[opt]i",
        avs_descale_create,
        (void *)(DESCALE_MODE_BILINEAR)
    );

    avs_add_function(
        env,
        "Debicubic",
        "c"
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
        "[opt]i",
        avs_descale_create,
        (void *)(DESCALE_MODE_BICUBIC)
    );

    avs_add_function(
        env,
        "Delanczos",
        "c"
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
        "[opt]i",
        avs_descale_create,
        (void *)(DESCALE_MODE_LANCZOS)
    );

    avs_add_function(
        env,
        "Despline16",
        "c"
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
        "[opt]i",
        avs_descale_create,
        (void *)(DESCALE_MODE_SPLINE16)
    );

    avs_add_function(
        env,
        "Despline36",
        "c"
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
        "[opt]i",
        avs_descale_create,
        (void *)(DESCALE_MODE_SPLINE36)
    );

    avs_add_function(env, "Despline64",
        "c"
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
        "[opt]i",
        avs_descale_create, (void*)(DESCALE_MODE_SPLINE64));

    avs_add_function(env, "Depoint",
        "c"
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
        "[opt]i",
        avs_descale_create, (void*)(DESCALE_MODE_POINT));

    avs_add_function(
        env,
        "Descale",
        "c"
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
        "[opt]i",
        avs_descale_create,
        NULL
    );

    return "Descale plugin";
}
