project('avisynth_headers', 'c', version: '3.7.5', meson_version: '>=1.0.0')

inc = include_directories('avs_core/include')

avisynth_headers_dep = declare_dependency(include_directories: inc)

meson.override_dependency('avisynth', avisynth_headers_dep)
