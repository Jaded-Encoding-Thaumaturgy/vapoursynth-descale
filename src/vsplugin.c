/* 
 * Copyright © 2017-2022 Frechdachs <frechdachs@rekt.cc>
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


#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vapoursynth/VapourSynth4.h>
#include <vapoursynth/VSHelper4.h>
#include "descale.h"
#include "plugin.h"


struct VSDescaleData
{
    bool initialized;
    pthread_mutex_t lock;

    VSNode *node;
    VSNode *ignore_mask_node;
    VSVideoInfo vi;

    struct DescaleData dd;
};


struct VSCustomKernelData
{
    const VSAPI *vsapi;
    VSFunction *custom_kernel;
    VSMap *cache;
};


static const char *VS_CC get_error(const char *funcname, const char *error) {
    const size_t flen = strlen(funcname);
    const size_t elen = strlen(error);

    char *out = malloc(flen + 2 + elen + 1);

    memcpy(out, funcname, flen);
    memcpy(out + flen, ": ", 2);
    memcpy(out + flen + 2, error, elen);
    memset(out + flen + 2 + elen, 0, 1);

    return out;
}

static const VSFrame *VS_CC descale_get_frame(int n, int activation_reason, void *instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi)
{
    struct VSDescaleData *d = (struct VSDescaleData *)instance_data;

    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frame_ctx);
        if (d->ignore_mask_node)
            vsapi->requestFrameFilter(n, d->ignore_mask_node, frame_ctx);

    } else if (activation_reason == arAllFramesReady) {
        if (!d->initialized) {
            pthread_mutex_lock(&d->lock);
            if (!d->initialized) {
                initialize_descale_data(&d->dd);
                d->initialized = true;
            }
            pthread_mutex_unlock(&d->lock);
        }

        const VSVideoFormat fmt = d->vi.format;
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frame_ctx);
        const VSFrame *ignore_mask = NULL;
        if (d->ignore_mask_node)
            ignore_mask = vsapi->getFrameFilter(n, d->ignore_mask_node, frame_ctx);

        VSFrame *intermediate = vsapi->newVideoFrame(&fmt, d->dd.dst_width, d->dd.src_height, NULL, core);
        VSFrame *dst = vsapi->newVideoFrame(&fmt, d->dd.dst_width, d->dd.dst_height, src, core);

        for (int plane = 0; plane < d->dd.num_planes; plane++) {
            int src_stride = vsapi->getStride(src, plane) / sizeof (float);
            int dst_stride = vsapi->getStride(dst, plane) / sizeof (float);
            const float *srcp = (const float *)vsapi->getReadPtr(src, plane);
            float *dstp = (float *)vsapi->getWritePtr(dst, plane);

            int imask_stride = 0;
            const unsigned char *imaskp = NULL;
            if (ignore_mask) {
                imask_stride = vsapi->getStride(ignore_mask, plane);
                imaskp = vsapi->getReadPtr(ignore_mask, plane);
            }

            if (d->dd.process_h && d->dd.process_v) {
                int intermediate_stride = vsapi->getStride(intermediate, plane) / sizeof (float);
                float *intermediatep = (float *)vsapi->getWritePtr(intermediate, plane);

                d->dd.dsapi.process_vectors(d->dd.dscore_h[plane && d->dd.subsampling_h], DESCALE_DIR_HORIZONTAL, d->dd.src_height >> (plane ? d->dd.subsampling_v : 0), src_stride, 0, intermediate_stride, srcp, NULL, intermediatep);
                d->dd.dsapi.process_vectors(d->dd.dscore_v[plane && d->dd.subsampling_v], DESCALE_DIR_VERTICAL, d->dd.dst_width >> (plane ? d->dd.subsampling_h : 0), intermediate_stride, 0, dst_stride, intermediatep, NULL, dstp);

            } else if (d->dd.process_h) {
                d->dd.dsapi.process_vectors(d->dd.dscore_h[plane && d->dd.subsampling_h], DESCALE_DIR_HORIZONTAL, d->dd.src_height >> (plane ? d->dd.subsampling_v : 0), src_stride, imask_stride, dst_stride, srcp, imaskp, dstp);

            } else if (d->dd.process_v) {
                d->dd.dsapi.process_vectors(d->dd.dscore_v[plane && d->dd.subsampling_v], DESCALE_DIR_VERTICAL, d->dd.src_width >> (plane ? d->dd.subsampling_h : 0), src_stride, imask_stride, dst_stride, srcp, imaskp, dstp);
            }
        }

        vsapi->freeFrame(intermediate);
        vsapi->freeFrame(src);
        vsapi->freeFrame(ignore_mask);

        return dst;
    }

    return NULL;
}


static void VS_CC descale_free(void *instance_data, VSCore *core, const VSAPI *vsapi)
{
    struct VSDescaleData *d = (struct VSDescaleData *)instance_data;

    vsapi->freeNode(d->node);
    vsapi->freeNode(d->ignore_mask_node);
    free(d->dd.params.post_conv);

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

    if (d->dd.params.mode == DESCALE_MODE_CUSTOM) {
        struct VSCustomKernelData *kd = (struct VSCustomKernelData *)d->dd.params.custom_kernel.user_data;
        vsapi->freeFunction(kd->custom_kernel);
        vsapi->freeMap(kd->cache);
        free(kd);
    }

    free(d);
}


static double custom_kernel_f(double x, void *user_data)
{
    struct VSCustomKernelData *kd = (struct VSCustomKernelData *)user_data;

    // Check cache first. Note that an undocumented `VSMap` requirement is that
    // keys must not start with a digit, hence the "k" prefix.
    unsigned long long y;
    memcpy(&y, &x, sizeof y);
    char cache_key[64];
    snprintf(cache_key, sizeof(cache_key), "k%llu", y);
    int err;
    double cached = kd->vsapi->mapGetFloat(kd->cache, cache_key, 0, &err);
    if (!err)
        return cached;

    VSMap *in = kd->vsapi->createMap();
    VSMap *out = kd->vsapi->createMap();
    kd->vsapi->mapSetFloat(in, "x", x, maReplace);
    kd->vsapi->callFunction(kd->custom_kernel, in, out);
    if (kd->vsapi->mapGetError(out)) {
        fprintf(stderr, "Descale: custom kernel error: %s.\n", kd->vsapi->mapGetError(out));
        kd->vsapi->freeMap(in);
        kd->vsapi->freeMap(out);
        return 0.0;
    }

    x = kd->vsapi->mapGetFloat(out, "val", 0, &err);
    if (err)
        x = (double)kd->vsapi->mapGetInt(out, "val", 0, &err);
    if (err) {
        fprintf(stderr, "Descale: custom kernel: The custom kernel function returned a value that is neither float nor int.");
        x = 0.0;
    }

    kd->vsapi->mapSetFloat(kd->cache, cache_key, x, maReplace);

    kd->vsapi->freeMap(in);
    kd->vsapi->freeMap(out);

    return x;
}


static void VS_CC descale_create(const VSMap *in, VSMap *out, void *user_data, VSCore *core, const VSAPI *vsapi)
{
    char *funcname;

    struct DescaleParams params = {
        .mode = (enum DescaleMode)user_data & (DESCALE_FLAG_SCALE - 1),
        .upscale = (enum DescaleMode)user_data & DESCALE_FLAG_SCALE,
    };

    switch(params.mode) {
        case DESCALE_MODE_BILINEAR:
            funcname = params.upscale ? "Bilinear" : "Debilinear"; 
            break;
        case DESCALE_MODE_BICUBIC:
            funcname = params.upscale ? "Bicubic" : "Debicubic"; 
            break;
        case DESCALE_MODE_LANCZOS:
            funcname = params.upscale ? "Lanczos" : "Delanczos"; 
            break;
        case DESCALE_MODE_SPLINE16:
            funcname = params.upscale ? "Spline16" : "Despline16"; 
            break;
        case DESCALE_MODE_SPLINE36:
            funcname = params.upscale ? "Spline36" : "Despline36"; 
            break;
        case DESCALE_MODE_SPLINE64:
            funcname = params.upscale ? "Spline64" : "Despline64"; 
            break;
        case DESCALE_MODE_CUSTOM:
            funcname = params.upscale ? "ScaleCustom" : "Decustom"; 
            break;
        default:
            vsapi->mapSetError(out, get_error("Descale", "Wrong API use!"));
            return;
    }

    VSNode *node = vsapi->mapGetNode(in, "src", 0, NULL);
    VSVideoInfo vi = *vsapi->getVideoInfo(node);

    if (!vsh_isConstantVideoFormat(&vi)) {
        vsapi->mapSetError(out, get_error(funcname, "Only constant format input is supported."));
        vsapi->freeNode(node);
        return;
    }

    if (vi.format.sampleType != stFloat || vi.format.bitsPerSample != 32) {
        vsapi->mapSetError(out, get_error(funcname, "Only float32 input is supported."));
        vsapi->freeNode(node);
        return;
    }

    VSFunction *custom_kernel = NULL;
    if (params.mode == DESCALE_MODE_CUSTOM) {
        custom_kernel = vsapi->mapGetFunction(in, "custom_kernel", 0, NULL);

        struct VSCustomKernelData *kd = malloc(sizeof (struct VSCustomKernelData));
        kd->vsapi = vsapi;
        kd->custom_kernel = custom_kernel;
        kd->cache = vsapi->createMap();

        params.custom_kernel.f = &custom_kernel_f;
        params.custom_kernel.user_data = kd;
    }

    int src_width = vi.width, src_height = vi.height;

    vi.width = vsapi->mapGetIntSaturated(in, "width", 0, NULL);
    vi.height = vsapi->mapGetIntSaturated(in, "height", 0, NULL);

    struct VSDescaleData d = {
        .initialized = false,
        .node = node,
        .vi = vi,
        .dd = {
            .src_width = src_width,
            .src_height = src_height,
            .dst_width = vi.width,
            .dst_height = vi.height,
            .subsampling_h = vi.format.subSamplingW,
            .subsampling_v = vi.format.subSamplingH,
            .num_planes = vi.format.numPlanes
        }
    };

    if (d.dd.dst_width % (1 << d.dd.subsampling_h) != 0) {
        vsapi->mapSetError(out, get_error(funcname, "Output width and output subsampling are not compatible."));
        vsapi->freeNode(d.node);
        return;
    }
    if (d.dd.dst_height % (1 << d.dd.subsampling_v) != 0) {
        vsapi->mapSetError(out, get_error(funcname, "Output height and output subsampling are not compatible."));
        vsapi->freeNode(d.node);
        return;
    }

    int err;

    d.ignore_mask_node = vsapi->mapGetNode(in, "ignore_mask", 0, &err);
    if (err) {
        d.ignore_mask_node = NULL;
    } else {
        params.has_ignore_mask = 1;
        const VSVideoInfo *mvi = vsapi->getVideoInfo(d.ignore_mask_node);
        if (mvi->format.sampleType != stInteger || mvi->format.bitsPerSample != 8) {
            vsapi->mapSetError(out, get_error(funcname, "Ignore mask must use 8 bit integer samples."));    // TODO improve this?
            vsapi->freeNode(d.node);
            vsapi->freeNode(d.ignore_mask_node);
            return;
        }

        if (mvi->format.numPlanes != d.vi.format.numPlanes
                || mvi->format.subSamplingH != d.vi.format.subSamplingH
                || mvi->format.subSamplingW != d.vi.format.subSamplingW
                || mvi->width != d.dd.src_width
                || mvi->height != d.dd.src_height
                || mvi->numFrames != d.vi.numFrames) {
            vsapi->mapSetError(out, get_error(funcname, "Ignore mask format must match clip format."));    // TODO improve this?
            vsapi->freeNode(d.node);
            vsapi->freeNode(d.ignore_mask_node);
            return;
        }
    }

    d.dd.shift_h = vsapi->mapGetFloat(in, "src_left", 0, &err);
    if (err)
        d.dd.shift_h = 0.0;

    d.dd.shift_v = vsapi->mapGetFloat(in, "src_top", 0, &err);
    if (err)
        d.dd.shift_v = 0.0;

    d.dd.active_width = vsapi->mapGetFloat(in, "src_width", 0, &err);
    if (err)
        d.dd.active_width = (double)(params.upscale ? d.dd.src_width : d.dd.dst_width);

    d.dd.active_height = vsapi->mapGetFloat(in, "src_height", 0, &err);
    if (err)
        d.dd.active_height = (double)(params.upscale ? d.dd.src_height : d.dd.dst_height);

    int border_handling = vsapi->mapGetIntSaturated(in, "border_handling", 0, &err);
    if (err)
        border_handling = 0;
    if (border_handling == 1)
        params.border_handling = DESCALE_BORDER_ZERO;
    else if (border_handling == 2)
        params.border_handling = DESCALE_BORDER_REPEAT;
    else
        params.border_handling = DESCALE_BORDER_MIRROR;

    enum DescaleOpt opt_enum;
    int opt = vsapi->mapGetIntSaturated(in, "opt", 0, &err);
    if (err)
        opt = 0;
    if (opt == 1)
        opt_enum = DESCALE_OPT_NONE;
    else if (opt == 2)
        opt_enum = DESCALE_OPT_AVX2;
    else
        opt_enum = DESCALE_OPT_AUTO;

    if (d.ignore_mask_node || params.upscale)
        opt_enum = DESCALE_OPT_NONE;

    if (d.dd.dst_width < 1) {
        vsapi->mapSetError(out, get_error(funcname, "width must be greater than 0."));
        vsapi->freeNode(d.node);
        vsapi->freeNode(d.ignore_mask_node);
        return;
    }

    if (d.dd.dst_height < 8) {
        vsapi->mapSetError(out, get_error(funcname, "Output height must be greater than or equal to 8."));
        vsapi->freeNode(d.node);
        vsapi->freeNode(d.ignore_mask_node);
        return;
    }

    if (!params.upscale && (d.dd.dst_width > d.dd.src_width || d.dd.dst_height > d.dd.src_height)) {
        vsapi->mapSetError(out, get_error(funcname, "Output dimension must be less than or equal to input dimension."));
        vsapi->freeNode(d.node);
        vsapi->freeNode(d.ignore_mask_node);
        return;
    }

    if (params.upscale && (d.dd.dst_width < d.dd.src_width || d.dd.dst_height < d.dd.src_height)) {
        vsapi->mapSetError(out, get_error(funcname, "Output dimension must be larger than or equal to input dimension."));
        vsapi->freeNode(d.node);
        vsapi->freeNode(d.ignore_mask_node);
        return;
    }

    d.dd.process_h = d.dd.dst_width != d.dd.src_width || d.dd.shift_h != 0.0 || d.dd.active_width != (double)d.dd.dst_width;
    d.dd.process_v = d.dd.dst_height != d.dd.src_height || d.dd.shift_v != 0.0 || d.dd.active_height != (double)d.dd.dst_height;

    if (params.mode == DESCALE_MODE_BICUBIC) {
        params.param1 = vsapi->mapGetFloat(in, "b", 0, &err);
        if (err)
            params.param1 = 0.0;

        params.param2 = vsapi->mapGetFloat(in, "c", 0, &err);
        if (err)
            params.param2 = 0.5;

        // If b != 0 Bicubic is not an interpolation filter, so force processing
        /*if (params.param1 != 0) {
            d.dd.process_h = true;
            d.dd.process_v = true;
        }*/
        // Leaving this check in would make it impossible to only descale a single dimension if this precondition is met.
        // If you want to force sampling use the force/force_h/force_v paramenter of the generic Descale filter.
    } else if (params.mode == DESCALE_MODE_LANCZOS || params.mode == DESCALE_MODE_CUSTOM) {
        params.taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err)
            params.taps = 3;

        if (params.taps < 1) {
            vsapi->mapSetError(out, get_error(funcname, "taps must be greater than 0."));
            vsapi->freeNode(d.node);
            vsapi->freeNode(d.ignore_mask_node);
            return;
        }
    }

    params.blur = vsapi->mapGetFloat(in, "blur", 0, &err);
    if (err)
        params.blur = 1.0;

    if (params.blur >= d.dd.src_width >> d.dd.subsampling_h || params.blur >= d.dd.src_height >> d.dd.subsampling_v || params.blur <= 0) {
        // We also need to ensure that the blur isn't smaller than 1 / support, but we can't know the exact support of the kernel here,
        vsapi->mapSetError(out, get_error(funcname, "blur parameter is out of bounds."));
        vsapi->freeNode(d.node);
        vsapi->freeNode(d.ignore_mask_node);
    }

    int force = vsapi->mapGetIntSaturated(in, "force", 0, &err);
    int force_h = vsapi->mapGetIntSaturated(in, "force_h", 0, &err);
    if (err)
        force_h = force;
    int force_v = vsapi->mapGetIntSaturated(in, "force_v", 0, &err);
    if (err)
        force_v = force;

    d.dd.process_h = d.dd.process_h || force_h;
    d.dd.process_v = d.dd.process_v || force_v;

    // Return the input clip if no processing is necessary
    if (!d.dd.process_h && !d.dd.process_v) {
        vsapi->mapSetNode(out, "clip", d.node, maReplace);
        vsapi->freeNode(d.node);
        vsapi->freeNode(d.ignore_mask_node);
        return;
    }

    if (d.dd.process_h && d.dd.process_v && d.ignore_mask_node) {
        vsapi->mapSetError(out, get_error(funcname, "Ignore mask is not supported when descaling along both axes."));
        vsapi->freeNode(d.node);
        vsapi->freeNode(d.ignore_mask_node);
        return;
    }

    if (params.upscale && d.ignore_mask_node) {
        vsapi->mapSetError(out, get_error(funcname, "Ignore mask is not supported when upscaling."));
        vsapi->freeNode(d.node);
        vsapi->freeNode(d.ignore_mask_node);
        return;
    }

    params.post_conv_size = vsapi->mapNumElements(in, "post_conv");
    if (params.post_conv_size == -1) {
        params.post_conv_size = 0;
    }

    if (params.post_conv_size) {
        if (params.post_conv_size % 2 != 1) {
            vsapi->mapSetError(out, get_error(funcname, "Post-convolution kernel must have odd length."));
            vsapi->freeNode(d.node);
            vsapi->freeNode(d.ignore_mask_node);
            return;
        }

        if ((d.dd.process_h && params.post_conv_size > 2 * vi.width + 1) || (d.dd.process_v && params.post_conv_size > 2 * vi.height + 1)) {
            vsapi->mapSetError(out, get_error(funcname, "Post-convolution kernel is too large, exceeds clip dimensions."));
            vsapi->freeNode(d.node);
            vsapi->freeNode(d.ignore_mask_node);
            return;
        }

        params.post_conv = calloc(params.post_conv_size, sizeof (double));
        for (int i = 0; i < params.post_conv_size; i++) {
            params.post_conv[i] = vsapi->mapGetFloat(in, "post_conv", i, &err);
        }
    }

    d.dd.dsapi = get_descale_api(opt_enum);
    pthread_mutex_init(&d.lock, NULL);

    struct VSDescaleData *data = malloc(sizeof d);
    *data = d;
    data->dd.params = params;
    VSFilterDependency deps[] = {{data->node, rpStrictSpatial}, {data->ignore_mask_node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, funcname, &data->vi, descale_get_frame, descale_free, fmParallel, deps, data->ignore_mask_node ? 2 : 1, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi)
{
    vspapi->configPlugin("tegaf.asi.xe", "descale", "Undo linear interpolation", VS_MAKE_VERSION(2, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

#define DESCALE_BASE_ARGS "src:vnode;width:int;height:int;"
#define DESCALE_COM_OUT_ARGS \
    "blur:float:opt;" \
    "post_conv:float[]:opt;" \
    "src_left:float:opt;src_top:float:opt;src_width:float:opt;src_height:float:opt;" \
    "border_handling:int:opt;" \
    "ignore_mask:vnode:opt;" \
    "force:int:opt;force_h:int:opt;force_v:int:opt;" \
    "opt:int:opt;", \
    "clip:vnode;"
#define DESCALE_ALL_ARGS DESCALE_BASE_ARGS DESCALE_COM_OUT_ARGS

#define DESCALE_REGISTER_FUNCTION(name_descale, name_scale, args, mode) \
    vspapi->registerFunction(name_descale, args, descale_create, (void *)(mode), plugin); \
    vspapi->registerFunction(name_scale, args, descale_create, (void *)(mode | DESCALE_FLAG_SCALE), plugin);

    DESCALE_REGISTER_FUNCTION("Debilinear", "Bilinear", DESCALE_ALL_ARGS, DESCALE_MODE_BILINEAR);

    DESCALE_REGISTER_FUNCTION("Debicubic", "Bicubic", DESCALE_BASE_ARGS "b:float:opt;" "c:float:opt;" DESCALE_COM_OUT_ARGS, DESCALE_MODE_BICUBIC);

    DESCALE_REGISTER_FUNCTION("Delanczos", "Lanczos", DESCALE_BASE_ARGS "taps:int:opt;" DESCALE_COM_OUT_ARGS, DESCALE_MODE_LANCZOS);

    DESCALE_REGISTER_FUNCTION("Despline16", "Spline16", DESCALE_ALL_ARGS, DESCALE_MODE_SPLINE16);

    DESCALE_REGISTER_FUNCTION("Despline36", "Spline36", DESCALE_ALL_ARGS, DESCALE_MODE_SPLINE36);

    DESCALE_REGISTER_FUNCTION("Despline64", "Spline64", DESCALE_ALL_ARGS, DESCALE_MODE_SPLINE64);

    DESCALE_REGISTER_FUNCTION("Decustom", "ScaleCustom", DESCALE_BASE_ARGS "custom_kernel:func;taps:int;" DESCALE_COM_OUT_ARGS, DESCALE_MODE_CUSTOM);

#undef DESCALE_REGISTER_FUNCTION
#undef DESCALE_BASE_ARGS
#undef DESCALE_COM_OUT_ARGS
#undef DESCALE_ALL_ARGS
}
