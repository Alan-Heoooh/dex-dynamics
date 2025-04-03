from setuptools import setup

install_requires = []


def read_requirements():
    with open("./requirements.txt", "r") as f:
        lines = [l.strip() for l in f.readlines()]
    install_requires = list(filter(None, lines))
    return install_requires


setup(
    name="dexwm",
    version="1.0.0",
    # install_requires=read_requirements(),
    py_modules=["dexwm"],
)
