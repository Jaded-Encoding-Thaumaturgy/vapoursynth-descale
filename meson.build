project('Descale', 'c', 'cpp',
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
    includedirs += ['avs_c_api_loader']
    sources += ['src/avsplugin.cpp', 'avs_c_api_loader/avs_c_api_loader.cpp']
endif

if libtype in ['vapoursynth', 'both']
    vs_dep = dependency('vapoursynth').partial_dependency(compile_args: true, includes: true)
    deps += [vs_dep]
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
    m_dep = cc.find_library('m', required: false)
    p_dep = cc.find_library('pthread')
    deps += [m_dep, p_dep]
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
    install: false
)

# Manual installation
install_dirs_to_use = []

if libtype in ['vapoursynth', 'both']
    vs_installdir = ''
    if host_machine.system() == 'windows'
        vs_installdir = join_paths(get_option('libdir'), 'vapoursynth')
    else
        vs_pkg_libdir = vs_dep.get_variable(pkgconfig: 'libdir', default_value: '')
        if vs_pkg_libdir == ''
            message('Warning: VapourSynth pkg-config "libdir" not found. Using default libdir for VapourSynth plugin installation.')
            vs_installdir = get_option('prefix') / get_option('libdir') / 'vapoursynth'
        else
            vs_installdir = join_paths(vs_pkg_libdir, 'vapoursynth')
        endif
    endif
    install_dirs_to_use += [vs_installdir]
endif

if libtype in ['avisynth', 'both']
    avs_installdir = ''
    if host_machine.system() == 'windows'
        avs_installdir = join_paths(get_option('libdir'), 'avisynth')
    else
        avs_pkg_dep = dependency('avisynth', required: false, method: 'pkg-config')
        avs_base_libdir = ''
        if avs_pkg_dep.found()
            avs_base_libdir = avs_pkg_dep.get_variable(pkgconfig: 'libdir', default_value: '')
        endif

        if avs_base_libdir != ''
            avs_installdir = join_paths(avs_base_libdir, 'avisynth')
            message('Using Avisynth pkg-config libdir for plugin installation: ' + avs_installdir)
        else
            message('Warning: Avisynth pkg-config "libdir" not found or Avisynth pkg-config itself not found. Using default libdir for Avisynth plugin installation.')
            avs_installdir = get_option('prefix') / get_option('libdir') / 'avisynth'
        endif
    endif
    install_dirs_to_use += [avs_installdir]
endif

if install_dirs_to_use.length() == 0 and libtype != 'none'
    error('No installation directory determined for libtype: ' + libtype)
endif

if install_dirs_to_use.length() > 0
    message('Setting up installation of descale module to: ' + ' | '.join(install_dirs_to_use))

    install_script_path = join_paths(meson.current_source_dir(), 'install_plugin.py')

    foreach installdir : install_dirs_to_use
        meson.add_install_script(
            python.full_path(),
            install_script_path,
            descale_module,
            installdir
        )
    endforeach
else
    if libtype != 'none'
        message('Descale module built but not installed as no valid install directories were determined for libtype: ' + libtype)
    endif
endif
