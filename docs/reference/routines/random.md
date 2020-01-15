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

[``Generator``](generator.html#numpy.random.Generator) can be used as a replacement for [``RandomState``](legacy.html#numpy.random.mtrand.RandomState). Both class
instances now hold a internal *BitGenerator* instance to provide the bit
stream, it is accessible as ``gen.bit_generator``. Some long-overdue API
cleanup means that legacy and compatibility methods have been removed from
[``Generator``](generator.html#numpy.random.Generator)

[RandomState](legacy.html#numpy.random.mtrand.RandomState) | [Generator](generator.html#numpy.random.Generator) | Notes
---|---|---
random_sample, | random | Compatible with [random.random](https://docs.python.org/dev/library/random.html#random.random)
rand |   |  
randint, | integers | Add an endpoint kwarg
random_integers |   |  
tomaxint | removed | Use integers(0, np.iinfo(np.int).max,endpoint=False)
seed | removed | Use [spawn](bit_generators/generated/numpy.random.SeedSequence.spawn.html#numpy.random.SeedSequence.spawn)

See *new-or-different* for more information

``` python
# As replacement for RandomState(); default_rng() instantiates Generator with
# the default PCG64 BitGenerator.
from numpy.random import default_rng
rg = default_rng()
rg.standard_normal()
rg.bit_generator
```

Something like the following code can be used to support both ``RandomState``
and ``Generator``, with the understanding that the interfaces are slightly
different

``` python
try:
    rg_integers = rg.integers
except AttributeError:
    rg_integers = rg.randint
a = rg_integers(1000)
```

Seeds can be passed to any of the BitGenerators. The provided value is mixed
via [``SeedSequence``](bit_generators/generated/numpy.random.SeedSequence.html#numpy.random.SeedSequence) to spread a possible sequence of seeds across a wider
range of initialization states for the BitGenerator. Here [``PCG64``](bit_generators/pcg64.html#numpy.random.pcg64.PCG64) is used and
is wrapped with a [``Generator``](generator.html#numpy.random.Generator).

``` python
from numpy.random import Generator, PCG64
rg = Generator(PCG64(12345))
rg.standard_normal()
```

## Introduction

The new infrastructure takes a different approach to producing random numbers
from the [``RandomState``](legacy.html#numpy.random.mtrand.RandomState) object.  Random number generation is separated into
two components, a bit generator and a random generator.

The *BitGenerator* has a limited set of responsibilities. It manages state
and provides functions to produce random doubles and random unsigned 32- and
64-bit values.

The [``random generator``](generator.html#numpy.random.Generator) takes the
bit generator-provided stream and transforms them into more useful
distributions, e.g., simulated normal random values. This structure allows
alternative bit generators to be used with little code duplication.

The [``Generator``](generator.html#numpy.random.Generator) is the user-facing object that is nearly identical to
[``RandomState``](legacy.html#numpy.random.mtrand.RandomState). The canonical method to initialize a generator passes a
[``PCG64``](bit_generators/pcg64.html#numpy.random.pcg64.PCG64) bit generator as the sole argument.

``` python
from numpy.random import default_rng
rg = default_rng(12345)
rg.random()
```

One can also instantiate [``Generator``](generator.html#numpy.random.Generator) directly with a *BitGenerator* instance.
To use the older [``MT19937``](bit_generators/mt19937.html#numpy.random.mt19937.MT19937) algorithm, one can instantiate it directly
and pass it to [``Generator``](generator.html#numpy.random.Generator).

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
