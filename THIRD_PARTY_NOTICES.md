# Third-party notices

The source code in this repository is released under the MIT licence (`LICENSE`); the weights, labels, figures and documentation are released under CC BY 4.0 (`LICENSE-DATA`). Parts of the repository derive from the software listed here, whose own licence terms continue to apply to those parts.

## robosat (Mapbox)

`pipeline/models.py` and `roofpedia_bern/models.py` adapt the ResNet-50 U-Net ("AlbuNet")
encoder-decoder, the `ConvRelu` and `DecoderBlock` building blocks, and the Lovasz hinge loss
from [robosat](https://github.com/mapbox/robosat), by way of Roofpedia. The architecture
itself follows Buslaev's TernausNet line of work; the implementation is robosat's.

```
MIT License

Copyright (c) 2018 Mapbox

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Roofpedia (NUS Urban Analytics Lab)

The tiling, training and extraction workflow follows
[Roofpedia](https://github.com/ualsg/Roofpedia) by Abraham Noah Wu and Filip Biljecki, which is
itself built on robosat. The Roofpedia repository publishes no licence file, so no licence grant
from its authors is relied on here. What this repository reuses is the robosat code above, under
the MIT terms reproduced above, together with the published description of the method, which is
credited by citation:

> Wu, A.N. and Biljecki, F. (2021). Roofpedia: Automatic mapping of green and solar roofs for an
> open roofscape registry and evaluation of urban sustainability. *Landscape and Urban Planning*,
> 214, 104167.

## PyTorch and torchvision

Pretrained ResNet-50 and ResNet-18 ImageNet weights are obtained at runtime from
[torchvision](https://github.com/pytorch/vision), which is distributed under the BSD 3-Clause
licence. Those weights are not redistributed in this repository.

## Data

The data used for training is covered by neither `LICENSE` nor `LICENSE-DATA` and keeps its own terms.

- SWISSIMAGE 10 cm, swissSURFACE3D, swissALTI3D and swissTLM3D: (c) swisstopo, open government
  data.
- Mapbox satellite tiles: (c) Mapbox and its data providers, subject to the Mapbox terms of
  service. No tiles and no API key are redistributed here.
- SWISSIMAGE RS for Bern: provided by the Swiss Territorial Data Lab (STDL) for research use and
  not redistributed here.
