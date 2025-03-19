# Binary files

This folder contains the `llamafiler` executable that is also included in BYOTA Docker images.
The executable is compiled by manually running the `build_llamafile.sh` script from the project root directory:

```
sh scripts/build_llamafile.sh
```

Llamafiler is part of [llamafile](https://github.com/Mozilla-Ocho/llamafile) and,
like llamafile itself, it is a [portable executable](https://github.com/jart/cosmopolitan)
that you can run out-of-the-box on many different systems. Just download the GGUF
file for an embedding model
(e.g. [all-MiniLM-L6-v2.F16.gguf](https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.F16.gguf))
and execute it as:

```
llamafiler -m all-MiniLM-L6-v2.F16.gguf
```
