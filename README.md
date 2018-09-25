# Witchcraft
Witchcraft is an engine providing a set of tools capable of encoding documents of various content types (flat, heirarchical, or series structured) into searchable latent vector space. Goal is to provide models capable of encoding images, videos, audio, text, tables, lists, and more.

## Warning
Right now this project is full of hot air. I'm slowly converting a mangled mess of test scripts into something actually usable. Demos of usage coming later and documentation is all in my head at the moment. Stay tuned

## Requirements

* Python 3.6+ is required
* Tensorflow 1.10 is required
* Tensorflow compiled with GPU support is optional but recommended

I have provided dockerfiles in `docker/` as a starter point to gives you all the dependencies you need to run a project using witchcraft

## Building
Witchcraft is a set of modules containing everything from NLP, tensorflow graphs and models, and audio/video analysis. There is a lot of work to do still, but the project structure is there.

To install as a system module or in your own project:
```
python3 setup.py install
```
