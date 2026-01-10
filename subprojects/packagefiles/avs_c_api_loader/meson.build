project('avs_c_api_loader', 'cpp', version: '1.3.0', meson_version: '>=1.0.0', default_options: ['cpp_std=c++20'])

avsinc = dependency('avisynth').partial_dependency(includes: true, compile_args: true)

subdir('src')

avs_c_api_loader_dep = declare_dependency(link_with: avs_c_api_loader, include_directories: include, dependencies: avsinc)

meson.override_dependency('avs_c_api_loader', avs_c_api_loader_dep)
