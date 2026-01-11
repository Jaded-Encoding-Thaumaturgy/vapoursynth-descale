project('Descale', 'c',
    default_options: ['buildtype=release', 'b_ndebug=if-release', 'c_std=c99', 'cpp_std=c++20'],
    meson_version: '>=0.56.0',
    version: '11'
)

add_global_arguments(['-D_XOPEN_SOURCE=700'], language: 'c')

cc = meson.get_compiler('c')

python = find_program('python3', 'python', required: true)

includedirs = ['include', 'src']

sources = ['src/descale.c']

resource_objects = []

libs = []

deps = []

libtype = get_option('libtype')

if libtype in ['vapoursynth', 'both']
    sources += ['src/vsplugin.c']
endif

if libtype in ['avisynth', 'both']
    add_languages('cpp', required: true, native: false)

    includedirs += ['avs_c_api_loader/src']
    sources += ['src/avsplugin.cpp', 'avs_c_api_loader/src/avs_c_api_loader.cpp']
endif

if libtype in ['vapoursynth', 'both']
    vs_dep = dependency('vapoursynth', required: false)
    if vs_dep.found()
        deps += [vs_dep.partial_dependency(compile_args: true, includes: true)]
    endif
endif

if libtype in ['avisynth', 'both']
    avs_dep = dependency('avisynth', required: false)
    if avs_dep.found()
        deps += [avs_dep.partial_dependency(compile_args: true, includes: true)]
    endif
endif

installdir = ''

if host_machine.system() == 'windows'
    if libtype in ['vapoursynth', 'both']
        m_dep = cc.find_library('m', required: false)
        p_dep = cc.find_library('winpthread')
        deps += [m_dep, p_dep]
        installdir = join_paths(get_option('libdir'), 'vapoursynth')
    else
        m_dep = []
        installdir = join_paths(get_option('libdir'), 'avisynth')
    endif

    # Windows Version Information
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

    target_apis_str = ''
    if libtype == 'vapoursynth'
        target_apis_str = 'VapourSynth'
    elif libtype == 'avisynth'
        target_apis_str = 'AviSynthPlus'
    else
        target_apis_str = 'VapourSynth and AviSynthPlus'
    endif

    conf_data.set_quoted('FILE_DESCRIPTION', 'Video/Image filter to undo upscaling for ' + target_apis_str)
    conf_data.set_quoted('INTERNAL_NAME', meson.project_name().to_lower())
    conf_data.set_quoted('ORIGINAL_FILENAME', 'lib' + meson.project_name().to_lower() + '.dll')
    conf_data.set_quoted('PRODUCT_NAME', meson.project_name())

    build_type_define = ''
    if get_option('buildtype') == 'debug'
        build_type_define = '-DPROJECT_DEBUG_BUILD'
        conf_data.set('PROJECT_DEBUG_BUILD', 1)
    else
        conf_data.set('PROJECT_DEBUG_BUILD', 0)
    endif

    windows_rc_configured = configure_file(
        input: 'src/descale.rc.in',
        output: 'descale.rc',
        configuration: conf_data
    )

    compiled_rc = windows.compile_resources(
        windows_rc_configured,
        include_directories: includedirs
    )
    resource_objects += compiled_rc
else
    if libtype in ['vapoursynth', 'both']
        m_dep = cc.find_library('m', required: false)
        p_dep = cc.find_library('pthread')
        deps += [m_dep, p_dep]

        vs_pkg_libdir = ''
        if vs_dep.found()
            vs_pkg_libdir = vs_dep.get_variable(pkgconfig: 'libdir', default_value: '')
        endif
        if vs_pkg_libdir != ''
            installdir = join_paths(vs_pkg_libdir, 'vapoursynth')
        else
            installdir = join_paths(get_option('libdir'), 'vapoursynth')
        endif
    else
        avs_pkg_libdir = ''
        if avs_dep.found()
            avs_pkg_libdir = avs_dep.get_variable(pkgconfig: 'libdir', default_value: '')
        endif

        if avs_pkg_libdir != ''
            installdir = join_paths(avs_pkg_libdir, 'avisynth')
        else
            installdir = join_paths(get_option('libdir'), 'avisynth')
        endif
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


descale_module = shared_module('descale', sources + resource_objects,
    dependencies: deps,
    include_directories: includedirs,
    link_with: libs,
    name_prefix: 'lib',
    install: true,
    install_dir: installdir
)

if libtype == 'both'
    if host_machine.system() == 'windows'
        suffix = 'dll'
    elif host_machine.system() == 'darwin'
        suffix = 'dylib'
    else
        suffix = 'so'
    endif

    filename = 'libdescale_avs.' + suffix

    # Determine AVS install path
    if host_machine.system() == 'windows'
        avs_install_dir = join_paths(get_option('libdir'), 'avisynth')
    else
        avs_pkg_libdir = ''
        if avs_dep.found()
            avs_pkg_libdir = avs_dep.get_variable(pkgconfig: 'libdir', default_value: '')
        endif

        if avs_pkg_libdir != ''
            avs_install_dir = join_paths(avs_pkg_libdir, 'avisynth')
        else
            avs_install_dir = get_option('prefix') / get_option('libdir') / 'avisynth'
        endif
    endif

    # Create the copy
    custom_target('descale_avs_copy',
        input: descale_module,
        output: filename,
        command: [
            python, '-c',
            'import shutil, sys; shutil.copy2(sys.argv[1], sys.argv[2])',
            '@INPUT@', '@OUTPUT@'
        ],
        install: true,
        install_dir: avs_install_dir
    )
endif
