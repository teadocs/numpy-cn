---
meta:
  - name: keywords
    content: 随机抽样
  - name: description
    content: Numpy的随机数例程使用 BitGenerator 和 Generator 的组合来生成伪随机数以创建序列，并使用这些序列从不同的统计分布中进行采样：
---

# 随机抽样 (``numpy.random``)

Numpy的随机数例程使用 BitGenerator 和 [``Generator``](generator.html#numpy.random.Generator) 的组合来生成伪随机数以创建序列，并使用这些序列从不同的统计分布中进行采样：

- BitGenerators：生成随机数的对象。这些通常是填充有32或64随机位序列的无符号整数字。
- 生成器：将来自BitGenerator的随机位序列转换为在指定间隔内遵循特定概率分布(如均匀、正态或二项式)的数字序列的对象。

从Numpy版本1.17.0开始，Generator可以使用许多不同的BitGenerator进行初始化。它暴露了许多不同的概率分布。有关更新的随机Numpy数例程的上下文，请参见[NEP 19](https://www.numpy.org/neps/nep-0019-rng-policy.html)。遗留的 [``RandomState``](legacy.html#numpy.random.mtrand.RandomState) 随机数例程仍然可用，但仅限于单个BitGenerator。

为了方便和向后兼容，单个[``RandomState``](legacy.html#numpy.random.mtrand.RandomState)实例的方法被导入到numpy.Random命名空间中，有关完整列表，请参阅[遗留随机生成](legacy.html#legacy)。

## 快速开始

默认情况下，[Generator](generator.html#numpy.random.Generator)使用[PCG64](bit_generators/pcg64.html#numpy.random.pcg64.PCG64)提供的位，
具有比传统mt19937
[``RandomState``](legacy.html#numpy.random.mtrand.RandomState)中的随机数字生成器更好的统计属性

``` python
# Uses the old numpy.random.RandomState
from numpy import random
random.standard_normal()
```
尽管随机值是由[`PCG64`](https://numpy.org/doc/1.17/reference/random/bit_generators/pcg64.html#numpy.random.pcg64.PCG64)生成的，但是[`Generator`](https://numpy.org/doc/1.17/reference/random/generator.html#numpy.random.Generator)可以直接替代RandomState。 [`Generator`](https://numpy.org/doc/1.17/reference/random/generator.html#numpy.random.Generator)包含一个BitGenerator的实例。 可以通过gen.bit_generator来访问。

```Python
# As replacement for RandomState(); default_rng() instantiates Generator with
# the default PCG64 BitGenerator.
from numpy.random import default_rng
rg = default_rng()
rg.standard_normal()
rg.bit_generator
```
Seeds可以传递给任何BitGenerator。 所提供的值通过[`SeedSequence`](https://numpy.org/doc/1.17/reference/random/bit_generators/generated/numpy.random.SeedSequence.html#numpy.random.SeedSequence)进行混合，以将可能的seeds序列分布在BitGenerator的更广泛的初始化状态中。 这里使用[`PCG64`](https://numpy.org/doc/1.17/reference/random/bit_generators/pcg64.html#numpy.random.pcg64.PCG64)并用[`Generator`](https://numpy.org/doc/1.17/reference/random/generator.html#numpy.random.Generator)包裹。

```python
from numpy.random import Generator, PCG64
rg = Generator(PCG64(12345))
rg.standard_normal()
```

## 介绍(`Introduction`)

新的基础架构采用了不同的方法来产生随机数
来自[``RandomState``](legacy.html＃numpy.random.mtrand.RandomState)对象。 随机数生成分为两个组件，一个`bit generator`和一个`random generator`。

*BitGenerator*具有有限的职责集。 它管理状态并提供产生随机双精度和随机无符号32位和64位值。

[`random generator`](generator.html＃numpy.random.Generator)将bit generator提供的流并将其转换为更有用的分布，例如模拟的正常随机值。 这种结构允许备用bit generator，几乎不需要代码重复。

[``Generator``](generator.html＃numpy.random.Generator)是面向用户的对象，几乎与
[``RandomState``](legacy.html＃numpy.random.mtrand.RandomState)相同。 初始化生成器的规范方法通过[``PCG64``](bit_generators/pcg64.html#numpy.random.pcg64.PCG64)  bit generator作为唯一的参数。

``` python
from numpy.random import default_rng
rg = default_rng(12345)
rg.random()
```


也可以使用 *BitGenerator* 实例直接实例化[`Generator`](generator.html#numpy.random.Generator)。
要使用较旧的[`MT19937`](bit_generators/mt19937.html#numpy.random.mt19937.MT19937)算法，可以直接实例化它
并将其传递给[`Generator`](generator.html#numpy.random.Generator)

``` python
from numpy.random import Generator, MT19937
rg = Generator(MT19937(12345))
rg.random()
```

### What’s New or Different

::: danger Warning

The Box-Muller method used to produce NumPy’s normals is no longer available
in [``Generator``](generator.html#numpy.random.Generator).  It is not possible to reproduce the exact random
values using Generator for the normal distribution or any other
distribution that relies on the normal such as the [``RandomState.gamma``](https://numpy.org/devdocs/reference/generated/numpy.random.mtrand.RandomState.gamma.html#numpy.random.mtrand.RandomState.gamma) or
[``RandomState.standard_t``](https://numpy.org/devdocs/reference/generated/numpy.random.mtrand.RandomState.standard_t.html#numpy.random.mtrand.RandomState.standard_t). If you require bitwise backward compatible
streams, use [``RandomState``](legacy.html#numpy.random.mtrand.RandomState).

:::

- The Generator’s normal, exponential and gamma functions use 256-step Ziggurat
methods which are 2-10 times faster than NumPy’s Box-Muller or inverse CDF
implementations.
- Optional ``dtype`` argument that accepts ``np.float32`` or ``np.float64``
to produce either single or double prevision uniform random variables for
select distributions
- Optional ``out`` argument that allows existing arrays to be filled for
select distributions
- [``random_entropy``](entropy.html#numpy.random.entropy.random_entropy) provides access to the system
source of randomness that is used in cryptographic applications (e.g.,
``/dev/urandom`` on Unix).
- All BitGenerators can produce doubles, uint64s and uint32s via CTypes
([``ctypes``](bit_generators/generated/numpy.random.pcg64.PCG64.ctypes.html#numpy.random.pcg64.PCG64.ctypes)) and CFFI ([``cffi``](bit_generators/generated/numpy.random.pcg64.PCG64.cffi.html#numpy.random.pcg64.PCG64.cffi)). This allows the bit generators
to be used in numba.
- The bit generators can be used in downstream projects via
[Cython](extending.html#randomgen-cython).
- [``integers``](https://numpy.org/devdocs/reference/generated/numpy.random.Generator.integers.html#numpy.random.Generator.integers) is now the canonical way to generate integer
random numbers from a discrete uniform distribution. The ``rand`` and
``randn`` methods are only available through the legacy [``RandomState``](legacy.html#numpy.random.mtrand.RandomState).
The ``endpoint`` keyword can be used to specify open or closed intervals.
This replaces both ``randint`` and the deprecated ``random_integers``.
- [``random``](https://numpy.org/devdocs/reference/generated/numpy.random.Generator.random.html#numpy.random.Generator.random) is now the canonical way to generate floating-point
random numbers, which replaces [``RandomState.random_sample``](https://numpy.org/devdocs/reference/generated/numpy.random.mtrand.RandomState.random_sample.html#numpy.random.mtrand.RandomState.random_sample),
*RandomState.sample*, and *RandomState.ranf*. This is consistent with
Python’s [``random.random``](https://docs.python.org/dev/library/random.html#random.random).
- All BitGenerators in numpy use [``SeedSequence``](bit_generators/generated/numpy.random.SeedSequence.html#numpy.random.SeedSequence) to convert seeds into
initialized states.

See [What’s New or Different](new-or-different.html#new-or-different) for a complete list of improvements and
differences from the traditional ``Randomstate``.

### Parallel Generation

The included generators can be used in parallel, distributed applications in
one of three ways:

- [SeedSequence spawning](parallel.html#seedsequence-spawn)
- [Independent Streams](parallel.html#independent-streams)
- [Jumping the BitGenerator state](parallel.html#parallel-jumped)

## Concepts

- [Random Generator](generator.html)
- [legacy mtrand](legacy.html)
- [Bit Generators](index.html)
- [Seeding and Entropy](parallel.html#seedsequence-spawn)

## Features

- [Parallel Applications](https://www.numpy.org/devdocs/reference/random/parallel.html)
  - [SeedSequence spawning](https://www.numpy.org/devdocs/reference/random/parallel.html#seedsequence-spawning)
  - [Independent Streams](https://www.numpy.org/devdocs/reference/random/parallel.html#independent-streams)
  - [Jumping the BitGenerator state](https://www.numpy.org/devdocs/reference/random/parallel.html#jumping-the-bitgenerator-state)
- [Multithreaded Generation](https://www.numpy.org/devdocs/reference/random/multithreading.html)
- [What’s New or Different](https://www.numpy.org/devdocs/reference/random/new-or-different.html)
- [Comparing Performance](https://www.numpy.org/devdocs/reference/random/performance.html)
  - [Recommendation](https://www.numpy.org/devdocs/reference/random/performance.html#recommendation)
  - [Timings](https://www.numpy.org/devdocs/reference/random/performance.html#timings)
  - [Performance on different Operating Systems](https://www.numpy.org/devdocs/reference/random/performance.html#performance-on-different-operating-systems)
- [Extending](https://www.numpy.org/devdocs/reference/random/extending.html)
  - [Numba](https://www.numpy.org/devdocs/reference/random/extending.html#numba)
  - [Cython](https://www.numpy.org/devdocs/reference/random/extending.html#cython)
  - [New Basic RNGs](https://www.numpy.org/devdocs/reference/random/extending.html#new-basic-rngs)
- [Reading System Entropy](https://www.numpy.org/devdocs/reference/random/entropy.html)

### Original Source

This package was developed independently of NumPy and was integrated in version
1.17.0. The original repo is at [https://github.com/bashtage/randomgen](https://github.com/bashtage/randomgen).
