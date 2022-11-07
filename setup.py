from setuptools import setup, find_packages
from pip._internal.req import parse_requirements
from pip._internal.network.session import PipSession

# install_reqs = parse_requirements('requirements.txt', session=PipSession())

setup(
    name='f2ai',
    version='0.0.2',
    description='',
    long_description='...',
    url='https://github.com/ai-excelsior/F2AI',
    long_description_content_type="text/x-rst",
    author='',
    license='MIT',
    zip_safe=False,
    packages=['aie_feast'],
)