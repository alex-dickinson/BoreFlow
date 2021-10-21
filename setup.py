"""BoreFlow

"""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]
# with open("optional_requirements.txt", "r") as fh:
#     optional_requirements = [line.strip() for line in fh]

def get_extra_requires(path, add_all=True):
    import re
    from collections import defaultdict
    with open(path) as fp:
        extra_deps = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith('#'):
                tags = set()
                if ':' in k:
                    k, v = k.split(':')
                    tags.update(vv.strip() for vv in v.split(','))
                tags.add(re.split('[<=>]', k)[0])
                for t in tags:
                    extra_deps[t].add(k)
        # add tag `all` at the end
        if add_all:
            extra_deps['all'] = set(vv for v in extra_deps.values() for vv in v)
    return extra_deps

setup(
    name="BoreFlow",
    version="0.0.1.dev",
    author="Alex Dickinson, Ben Mather and Mark Ireland",
    author_email="nad38@cantab.ac.uk",
    description="WHAT IS THE REASON YOU BUILD THIS PROJECT AND WHAT IT DOES",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alex-dickinson/BoreFlow",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific"
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
	extras_require=get_extra_requires('optional_requirements.txt'),
)