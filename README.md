# Witchcraft
Engine that is capable of converting documents of arbiturary, heirarchical content (wishlist to support lists, tables, paragraphs, code, photo, audio, and video) into searchable, latent space.

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
