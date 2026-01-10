project('Descale', 'c',
    default_options: ['buildtype=release', 'b_ndebug=if-release', 'c_std=c99', 'cpp_std=c++20'],
    meson_version: '>=0.56.0',
    version: '11'
)

# Languages and compile flags

libtype = get_option('libtype')

use_cpp = libtype in ['avisynth', 'both']
languages = ['c']
if use_cpp
    add_languages('cpp', required: true, native: false)
    languages += 'cpp'
endif

foreach lang : languages
  if meson.get_compiler(lang).get_argument_syntax() == 'msvc'
    error('MSVC and MSVC-like compilers like clang-cl are not supported. Please use a GCC-like toolchain.')
  endif
endforeach

add_global_arguments(['-D_XOPEN_SOURCE=700'], language: languages)

cc = meson.get_compiler('c')

# Sources and dependencies

sources = ['src/descale.c']
includedirs = ['include', 'src']
libs = []
deps = []

if libtype in ['avisynth', 'both']
    sources += ['src/avsplugin.cpp']
    deps += dependency('avs_c_api_loader', fallback: 'avs_c_api_loader')
endif

if libtype in ['vapoursynth', 'both']
    sources += ['src/vsplugin.c']
    vs = dependency('vapoursynth').partial_dependency(compile_args: true, includes: true)
    deps += [vs]
endif

if host_machine.system() == 'windows'
    # Statically link things on Windows.
    m_dep = cc.find_library('m', static: true, required: false)
    p_dep = cc.find_library('winpthread', static: true)
    deps += [m_dep, p_dep]
else
    m_dep = cc.find_library('m', required: false)
    p_dep = cc.find_library('pthread')
    deps += [m_dep, p_dep]
endif

# Architecture-dependent logic

if host_machine.cpu_family().startswith('x86')
    add_project_arguments('-DDESCALE_X86', '-mfpmath=sse', '-msse2', language : languages)

    sources += ['src/x86/cpuinfo_x86.c']

    libs += static_library('descale_avx2', 'src/x86/descale_avx2.c',
                dependencies: [m_dep],
                c_args: ['-mavx2', '-mfma'],
                pic: true,
                include_directories: includedirs
            )
endif

# Windows version information

if host_machine.system() == 'windows'
    windows = import('windows')

    project_version_parts = meson.project_version().split('.')
    version_major = project_version_parts[0].to_int()
    version_minor = project_version_parts.length() > 1 ? project_version_parts[1].to_int() : 0
    version_patch = project_version_parts.length() > 2 ? project_version_parts[2].to_int() : 0
    version_build = project_version_parts.length() > 3 ? project_version_parts[3].to_int() : 0

    conf_data = configuration_data()
    conf_data.set('VERSION_MAJOR', version_major)
    conf_data.set('VERSION_MINOR', version_minor)
    conf_data.set('VERSION_PATCH', version_patch)
    conf_data.set('VERSION_BUILD', version_build)

    target_apis_strs = {
        'vapoursynth': 'VapourSynth',
        'avisynth': 'AviSynthPlus',
        'both': 'VapourSynth and AviSynthPlus',
    }

    conf_data.set_quoted('FILE_DESCRIPTION', 'Video/Image filter to undo upscaling for ' + target_apis_strs[libtype])
    conf_data.set_quoted('INTERNAL_NAME', meson.project_name().to_lower())
    conf_data.set_quoted('ORIGINAL_FILENAME', 'lib' + meson.project_name().to_lower() + '.dll')
    conf_data.set_quoted('PRODUCT_NAME', meson.project_name())
    conf_data.set10('PROJECT_DEBUG_BUILD', get_option('buildtype') == 'debug')

    windows_rc_configured = configure_file(
        input: 'src/descale.rc.in',
        output: 'descale.rc',
        configuration: conf_data
    )
    compiled_rc = windows.compile_resources(
        windows_rc_configured,
        include_directories: includedirs
    )
    sources += compiled_rc
endif

# Install path logic

if host_machine.system() == 'windows'
    vsinstalldir = join_paths(get_option('libdir'), 'vapoursynth')
    avsinstalldir = join_paths(get_option('libdir'), 'avisynth')
else
    if libtype in ['vapoursynth', 'both']
        vsinstalldir = join_paths(vs.get_variable(pkgconfig: 'libdir'), 'vapoursynth')
    endif

    if libtype in ['avisynth', 'both']
        avs = dependency('avisynth', required: false)
        avs_libdir = ''
        if avs.found()
            avs_libdir = avs.get_variable(pkgconfig: 'libdir', default_value: '')
        endif

        avsinstalldir = join_paths(avs_libdir == '' ? get_option('libdir') : avs_libdir, 'avisynth')
    endif
endif

if libtype in ['vapoursynth', 'both']  # I'm not sure if it is possible to install
                                       # the _same_ file to multiple directories with meson
    installdir = vsinstalldir
else
    installdir = avsinstalldir
endif

# Descale plugin

shared_module('descale', sources,
    dependencies: deps,
    include_directories: includedirs,
    link_with: libs,
    name_prefix: 'lib',
    install: true,
    install_dir: installdir
)
