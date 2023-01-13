from setuptools import setup, find_namespace_packages
from pip._internal.req import parse_requirements
from pip._internal.network.session import PipSession

install_requirements = parse_requirements("requirements.txt", session=PipSession())

setup(
    name="f2ai",
    version="0.0.4",
    description="A Feature Store tool focus on making retrieve features easily in machine learning.",
    url="https://github.com/ai-excelsior/F2AI",
    author="上海半见",
    license="MIT",
    zip_safe=False,
    packages=find_namespace_packages(exclude=["docs", "tests", "tests.*", "use_cases", "*.egg-info"]),
    install_requires=[str(x.requirement) for x in install_requirements],
    entry_points={"console_scripts": ["f2ai = f2ai.cmd:main"]},
)
