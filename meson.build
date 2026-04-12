project('Descale', 'c',
    default_options: ['buildtype=release', 'b_ndebug=if-release', 'c_std=c99'],
    meson_version: '>=0.64.0',
    version: '12'
)

cc = meson.get_compiler('c')

py = import('python').find_installation(pure: false)

r = run_command(
    py,
    '-c',
    'import vapoursynth as vs; print(vs.get_include())',
    check: true,
)
inc_vs = include_directories(r.stdout().strip())

common_c_args = []
if cc.get_id() != 'msvc'
    common_c_args += ['-D_XOPEN_SOURCE=700']
else
    common_c_args += ['-D_CRT_SECURE_NO_WARNINGS']
endif

add_project_arguments(common_c_args, language: 'c')

# Include directories and source files
includedirs = include_directories('include', 'src')
sources = ['src/descale.c']
libs = []
deps = []
extra_inc = []

libtype = get_option('libtype')

# Sources

if libtype in ['vapoursynth', 'both']
    sources += ['src/vsplugin.c']
endif

if libtype in ['avisynth', 'both']
    sources += ['src/avsplugin.c']
endif

# AviSynth dependency
if libtype in ['avisynth', 'both']
    if host_machine.system() == 'windows'
        deps += [cc.find_library('AviSynth', dirs: meson.current_source_dir())]
    else
        deps += [dependency('avisynth')]
    endif
endif

if host_machine.system() == 'windows'
    if cc.get_id() != 'msvc'
        # msys2
        m_dep = cc.find_library('m', static: true, required: false)
        deps += [m_dep, cc.find_library('winpthread', static: true)]
    else
        # MSVC
        m_dep = []
        found_pthreads = false
        foreach p : ['pthreadVC3', 'pthreadVC2', 'pthread', 'winpthread']
            p_dep = cc.find_library(p, required: false)
            if p_dep.found()
                deps += [p_dep]
                found_pthreads = true
                break
            endif
        endforeach
        if not found_pthreads
            extra_inc += include_directories('src/compat')
        endif
    endif
else
    m_dep = cc.find_library('m', required: false)
    deps += [m_dep, dependency('threads')]
endif

# Architecture-dependent logic
if host_machine.cpu_family().startswith('x86')
    add_project_arguments('-DDESCALE_X86', language : 'c')

    avx2_args = []
    if cc.get_id() == 'msvc'
        avx2_args += ['/arch:AVX2']
    else
        avx2_args += ['-mavx2', '-mfma']
    endif

    sources += ['src/x86/cpuinfo_x86.c']

    libs += static_library('descale_avx2', 'src/x86/descale_avx2.c',
                dependencies: deps,
                c_args: avx2_args,
                pic: true,
                include_directories: [includedirs, inc_vs, extra_inc]
            )
endif

# Install path logic
if libtype in ['vapoursynth', 'both']
    installdir = py.get_install_dir() / 'vapoursynth/plugins'
else
    installdir = join_paths(get_option('libdir'), 'avisynth')
endif


shared_module('descale', sources,
    dependencies: deps,
    include_directories: [includedirs, inc_vs, extra_inc],
    link_with: libs,
    name_prefix: 'lib',
    install: true,
    install_dir: installdir,
    gnu_symbol_visibility: 'hidden'
)
