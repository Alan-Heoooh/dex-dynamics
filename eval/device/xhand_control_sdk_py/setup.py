from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import platform

script_dir = os.path.dirname(os.path.realpath(__file__))
xhandcontrol_include_dir = os.path.join(script_dir, "include")
xhandcontrol_library_dir = os.path.join(script_dir, "lib")

# 扩展模块配置
ext_module = [
    Pybind11Extension(
        "xhand_controller.xhand_control",
        ["src/xhand_control.cpp"],
        libraries=["xhand_control", "soem"],  # 包含主.so文件和依赖的.so文件
        library_dirs=[xhandcontrol_library_dir],  # 使用绝对路径指定库目录
        include_dirs=[xhandcontrol_include_dir],  # 使用绝对路径指定头文件目录
        extra_link_args=["-Wl,-rpath,$ORIGIN/lib"],
    ),
]

# 所需的 .so 文件
so_files = [
    os.path.join('lib', 'libsoem.so'),
    os.path.join('lib', 'libxhand_control.so')
]

# 如果目标平台是 Linux 或 Darwin，并需要设置库路径
extra_link_args = []
if platform.system() in ["Linux", "Darwin"]:
    extra_link_args.append(f"-Wl,-rpath,{xhandcontrol_library_dir}")

setup(
    name="xhand_controller",
    version="0.10.3",
    author="Your Name",
    author_email="your.email@example.com",
    description="Python bindings for XHandControl library",
    ext_modules=ext_module,
    package_data={"xhand_controller": ["*.so", "*.dll", "*.dylib"]},  # 包含动态库文件
    cmdclass={"build_ext": build_ext},
    packages=["xhand_controller"],
    data_files=[('/usr/local/lib', so_files)],  # 将 .so 文件复制到 /usr/local/lib
    include_package_data=True,
    install_requires=[
        "pybind11==2.13.1",
    ],
)
