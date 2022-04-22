# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from setuptools import setup, find_packages

setup(
    name="axonn",
    version="0.1.0",
    description="A parallel library for extreme-scale deep learning",
    long_description="""An asynchronous, message-driven parallel framework for
        extreme-scale deep learning""",
    url="https://github.com/hpcgroup/axonn",
    author="Siddharth Singh, Abhinav Bhatele",
    author_email="ssingh37@umd.edu, bhatele@cs.umd.edu",
    classifiers=["Development Status :: 2 - Pre-Alpha"],
    keywords="deep learning, distributed computing, parallel computing",
    packages=find_packages(),
    install_requires=["torch", "mpi4py"],
)
