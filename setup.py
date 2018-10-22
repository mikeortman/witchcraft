from setuptools import setup

setup(
    name="witchcraft",
    description="https://github.com/mikeortman/witchcraft",
    license="GNU General Public License v3.0",
    version="0.0.1",
    author="Mike Ortman",
    author_email="mikeortman@gmail.com",
    url="https://github.com/mikeortman/witchcraft",
    packages=["witchcraft",
              "witchcraft.nlp",
              "witchcraft.nlp.protos",
              "witchcraft.ml",
              "witchcraft.ml.protos",
              "witchcraft.util",
              "projects",
              "projects.naughty",
              "projects.naughty.protos"],
    install_requires=[
        'spacy', 'tensorflow'
    ],
    setup_requires=[
        'pytest-runner', 'pytest-pylint', 'pytest-mypy', 'mypy-protobuf', 'protobuf'
    ],
    tests_require=["mypy", "pylint", "pytest", "spacy"]
)
