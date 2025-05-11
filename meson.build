project('Descale', 'c', 'cpp',
    default_options: ['buildtype=release', 'b_ndebug=if-release', 'c_std=c99', 'cpp_std=c++20'],
    meson_version: '>=0.56.0',
    version: '11'
)

add_global_arguments(['-D_XOPEN_SOURCE=700'], language: 'c')

cc = meson.get_compiler('c')

includedirs = ['include', 'src']

sources = ['src/descale.c']

libs = []

deps = []

libtype = get_option('libtype')

if libtype in ['vapoursynth', 'both']
    sources += ['src/vsplugin.c']
endif

if libtype in ['avisynth', 'both']
    includedirs += ['avs_c_api_loader']
    sources += ['src/avsplugin.cpp', 'avs_c_api_loader/avs_c_api_loader.cpp']
endif

if libtype in ['vapoursynth', 'both']
    vs = dependency('vapoursynth').partial_dependency(compile_args: true, includes: true)
    deps += [vs]
endif
if host_machine.system() == 'windows'
    if cc.get_id() == 'gcc' or cc.get_id() == 'clang'
        # Statically link things on Windows.
        m_dep = cc.find_library('m', static: true, required: false)
        p_dep = cc.find_library('winpthread', static: true)
        deps += [m_dep, p_dep]
    else
        m_dep = []
    endif
    if libtype in ['vapoursynth', 'both']  # I'm not sure if it is possible to install
                                           # the _same_ file to multiple directories with meson
        installdir = join_paths(get_option('libdir'), 'vapoursynth')
    else
        installdir = join_paths(get_option('libdir'), 'avisynth')
    endif
else
    m_dep = cc.find_library('m', required: false)
    p_dep = cc.find_library('pthread')
    deps += [m_dep, p_dep]
    if libtype in ['vapoursynth', 'both']
        installdir = join_paths(vs.get_variable(pkgconfig: 'libdir'), 'vapoursynth')
    else
        installdir = join_paths(avs.get_variable(pkgconfig: 'libdir'), 'avisynth')
    endif
endif

if host_machine.cpu_family().startswith('x86')
    x86_args = ['-DDESCALE_X86', '-mfpmath=sse', '-msse2']

    add_project_arguments(x86_args, language : 'c')

    if libtype in ['avisynth', 'both']
        add_project_arguments(x86_args, language : 'cpp')
    endif

    sources += ['src/x86/cpuinfo_x86.c']

    libs += static_library('descale_avx2', 'src/x86/descale_avx2.c',
                dependencies: [m_dep],
                c_args: ['-mavx2', '-mfma'],
                pic: true,
                include_directories: includedirs
            )
endif


shared_module('descale', sources,
    dependencies: deps,
    include_directories: includedirs,
    link_with: libs,
    name_prefix: 'lib',
    install: true,
    install_dir: installdir
)
