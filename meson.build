project('Descale', 'c',
    default_options: ['buildtype=release', 'b_ndebug=if-release', 'c_std=c99'],
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

# Sources

if libtype in ['vapoursynth', 'both']
    sources += ['src/vsplugin.c']
endif

if libtype in ['avisynth', 'both']
    sources += ['src/avsplugin.c']
endif

# Dependencies

if libtype in ['vapoursynth', 'both']
    vs = dependency('vapoursynth').partial_dependency(compile_args: true, includes: true)
    deps += [vs]
endif

if libtype in ['avisynth', 'both']
    if host_machine.system() == 'windows'
        deps += [cc.find_library('AviSynth', dirs: meson.current_source_dir())]
    else
        avs = dependency('avisynth')
        deps += [avs]
    endif
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
    add_project_arguments('-DDESCALE_X86', '-mfpmath=sse', '-msse2', language : 'c')

    sources += ['src/x86/cpuinfo_x86.c']

    libs += static_library('descale_avx2', 'src/x86/descale_avx2.c',
                dependencies: [m_dep],
                c_args: ['-mavx2', '-mfma'],
                pic: true,
                include_directories: includedirs
            )
endif

# Install path logic

if host_machine.system() == 'windows'
    if libtype in ['vapoursynth', 'both']  # I'm not sure if it is possible to install
                                           # the _same_ file to multiple directories with meson
        installdir = join_paths(get_option('libdir'), 'vapoursynth')
    else
        installdir = join_paths(get_option('libdir'), 'avisynth')
    endif
else
    if libtype in ['vapoursynth', 'both']
        installdir = join_paths(vs.get_variable(pkgconfig: 'libdir'), 'vapoursynth')
    else
        installdir = join_paths(avs.get_variable(pkgconfig: 'libdir'), 'avisynth')
    endif
endif


shared_module('descale', sources,
    dependencies: deps,
    include_directories: includedirs,
    link_with: libs,
    name_prefix: 'lib',
    install: true,
    install_dir: installdir
)
