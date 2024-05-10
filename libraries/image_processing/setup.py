from setuptools import setup, Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
import os
import numpy

class CustomBuildExtCommand(build_ext):
    def build_extensions(self):
        nvcc = '/usr/bin/nvcc'
        for ext in self.extensions:
            cuda_sources = [s for s in ext.sources if s.endswith('.cu')]
            for source in cuda_sources:
                output_file = os.path.splitext(source)[0] + '.o'
                command = [nvcc, '-c', source, '-o', output_file, '-Xcompiler', '-fPIC', '-std=c++11']
                print("Running command:", " ".join(command))  # Debug output of the command
                try:
                    self.spawn(command)
                    ext.extra_objects.append(output_file)
                    ext.sources.remove(source)
                except Exception as e:
                    print(f"Error compiling {source} with nvcc")
                    raise RuntimeError("CUDA compilation failed") from e
        super().build_extensions()

extensions = [
    Extension(
        "image_processing",
        sources=["image_processing.pyx", "image_processing_wrapper.cpp", "image_processing.cu"],
        include_dirs=[numpy.get_include(), '/usr/include', '/usr/include/x86_64-linux-gnu'],
        library_dirs=['/usr/lib/x86_64-linux-gnu'],
        libraries=['cudart'],
        extra_compile_args=['-std=c++11', '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
        language='c++'
    )
]

setup(
    name="ImageProcessing",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    cmdclass={'build_ext': CustomBuildExtCommand},
    zip_safe=False,
)