# NumPy 的内部

- [NumPy C Code Explanations](internals.code-explanations.html)
  - [Memory model](internals.code-explanations.html#memory-model)
  - [Data-type encapsulation](internals.code-explanations.html#data-type-encapsulation)
  - [N-D Iterators](internals.code-explanations.html#n-d-iterators)
  - [Broadcasting](internals.code-explanations.html#broadcasting)
  - [Array Scalars](internals.code-explanations.html#array-scalars)
  - [Indexing](internals.code-explanations.html#indexing)
    - [Advanced indexing](internals.code-explanations.html#advanced-indexing)
  - [Universal Functions](internals.code-explanations.html#universal-functions)
    - [Setup](internals.code-explanations.html#setup)
    - [Function call](internals.code-explanations.html#function-call)
      - [One Loop](internals.code-explanations.html#one-loop)
      - [Strided Loop](internals.code-explanations.html#strided-loop)
      - [Buffered Loop](internals.code-explanations.html#buffered-loop)
    - [Final output manipulation](internals.code-explanations.html#final-output-manipulation)
    - [Methods](internals.code-explanations.html#methods)
      - [Setup](internals.code-explanations.html#id1)
      - [Reduce](internals.code-explanations.html#reduce)
      - [Accumulate](internals.code-explanations.html#accumulate)
      - [Reduceat](internals.code-explanations.html#reduceat)
- [Memory Alignment](alignment.html)
  - [Numpy Alignment Goals](alignment.html#numpy-alignment-goals)
  - [Variables in Numpy which control and describe alignment](alignment.html#variables-in-numpy-which-control-and-describe-alignment)
  - [Consequences of alignment](alignment.html#consequences-of-alignment)

## NumPy 数组的内部组织结构

这有助于您了解一些有关如何处理numpy数组的知识，以帮助您更好地理解numpy。本节将不会详细介绍。那些希望了解全部细节的人可以参考Travis Oliphant的书“ NumPy指南”。

NumPy数组由两个主要组件组成，原始数组数据（从现在开始，称为数据缓冲区），以及有关原始数组数据的信息。人们通常认为数据缓冲区是C或Fortran中的数组，这是包含固定大小的数据项的连续（固定）内存块。NumPy还包含一组重要的数据，这些数据描述了如何解释数据缓冲区中的数据。此额外信息包含（除其他事项外）：

1. 基本数据元素的大小（以字节为单位）
1. 数据缓冲区中数据的起始位置（相对于数据缓冲区起始位置的偏移量）。
1. 尺寸数和每个尺寸的大小
1. 每个维度的元素之间的分隔（“跨度”）。这不必是元素大小的倍数
1. 数据的字节顺序（可能不是本机字节顺序）
1. 缓冲区是否为只读
1. 有关基本数据元素解释的信息（通过dtype对象）。基本数据元素可以像int或float一样简单，也可以是复合对象（例如，类似于struct的对象），固定字符字段或Python对象指针。
1. 数组是解释为C顺序还是Fortran顺序。

这种布置允许非常灵活地使用阵列。它允许的一件事是对元数据的简单更改即可更改数组缓冲区的解释。更改数组的字节顺序是一个简单的更改，不涉及数据的重新排列。可以很容易地更改数组的形状，而无需更改数据缓冲区中的任何内容或进行任何数据复制

除其他外，可以创建一个新的数组元数据对象，该对象使用相同的数据缓冲区来创建该数据缓冲区的新视图，该视图对缓冲区的解释不同（例如，形状，偏移量，字节顺序，跨步等），但共享相同的数据字节。numpy中的许多操作就是这样做的，例如切片。其他操作（例如转置）不会在数组中四处移动数据元素，而是会更改有关形状和步幅的信息，以使数组的索引发生变化，但数组中的数据不会移动。

通常，这些新版本的数组元数据和相同的数据缓冲区是数据缓冲区的新“视图”。有一个不同的ndarray对象，但是它使用相同的数据缓冲区。这就是为什么如果一个人真的想创建数据缓冲区的新的独立副本，则必须通过使用.copy（）方法来强制进行复制。

数组中的新视图意味着数据缓冲区的对象引用计数增加。如果仍然保留原始数组对象的其他视图，则仅删除原始数组对象将不会删除该数据缓冲区。

## 多维数组索引顺序问题

索引多维数组的正确方法是什么？在得出关于索引多维数组的一种真实方法的结论之前，有必要了解为什么这是一个令人困惑的问题。本节将尝试详细解释numpy索引的工作方式，为什么我们采用对图像的约定，以及何时适合采用其他约定。

首先要了解的是，索引二维数组有两种冲突的约定。矩阵表示法使用第一个索引指示正在选择的行，使用第二个索引指示选择的列。这与图像的几何定向惯例相反，人们通常认为第一个索引代表x位置（即列），第二个索引代表y位置（即行）。仅此一项就造成了很多混乱。面向矩阵的用户和面向图像的用户期望在索引方面有两件事。

要了解的第二个问题是索引如何与数组在内存中存储的顺序相对应。在Fortran中，第一个索引是在存储于内存中的二维数组中移动时变化最快的索引。如果采用矩阵约定进行索引，则这意味着矩阵一次存储一列（因为第一个索引随着更改而移动到下一行）。因此，Fortran被认为是列主要语言。C有相反的约定。在C语言中，最后一个索引随着存储在内存中的数组的移动而变化最快。因此，C是行专用语言。矩阵按行存储。请注意，在这两种情况下，都假定使用了索引的矩阵约定，即对于Fortran和C，第一个索引是行。

但这不是观察它的唯一方法。假设其中有一个大型的二维数组（图像或矩阵）存储在数据文件中。假设数据是按行而不是按列存储的。如果我们要保留索引约定（无论是矩阵还是图像），这意味着取决于我们使用的语言，如果将数据读入内存以保留索引约定，我们可能会被迫重新排序数据。例如，如果我们将行排序的数据读入内存而不进行重新排序，则它将匹配C的矩阵索引约定，但不匹配Fortran。相反，它将与Fortran的图像索引约定匹配，但与C不匹配。对于C，如果使用的是按行顺序存储的数据，并且希望保留图像索引约定，则在读入内存时必须对数据重新排序。

最后，您对Fortran或C进行的操作取决于哪个更重要，而不是对数据重新排序或保留索引约定。对于大图像，对数据进行重新排序可能会非常昂贵，并且通常会颠倒索引约定以避免这种情况。

numpy的情况使此问题更加复杂。numpy数组的内部机制足够灵活，可以接受索引的任何排序。可以通过处理数组的内部步幅信息来简单地对索引进行重新排序，而根本不需要对数据进行重新排序。NumPy将知道如何在不移动数据的情况下将新索引顺序映射到数据。

因此，如果这是真的，为什么不选择与您最期望的匹配的索引顺序？特别是，为什么不定义按行排序的图像以使用图像约定？（这有时被称为Fortran约定与C约定，因此以numpy进行数组排序的'C'和'FORTRAN'排序选项。）这样做的缺点是潜在的性能损失。顺序访问数据是很常见的，既可以在数组操作中隐式地进行，也可以通过循环遍历图像的行来显式地进行。完成此操作后，将以非最佳顺序访问数据。随着第一个索引的增加，实际上发生的是顺序访问内存中间隔很远的元素，通常访问内存的速度较差。例如，对于定义为使im [0，10]表示x = 0处的值的二维图像“ im”，Y = 10。为了与通常的Python行为保持一致，则im [0]将在x = 0处表示一列。但是，由于数据按行顺序存储，因此该数据将散布在整个阵列上。尽管numpy的索引具有灵活性，但它无法真正说明由于数据顺序使基本操作效率低下或获取连续子数组仍然很尴尬的事实（例如，第一行的im [：，0]与im [ 0]），因此不能使用诸如im中的row这样的惯用语；for im in col可以工作，但不会产生连续的列数据。它不能真正地说明由于数据顺序使基本操作效率低下或获得连续子数组仍然很尴尬的事实（例如，第一行的im [：，0]与im [0]），因此可以不要在im中使用诸如for的成语；for im in col可以工作，但不会产生连续的列数据。它不能真正地说明由于数据顺序使基本操作效率低下或获得连续子数组仍然很尴尬的事实（例如，第一行的im [：，0]与im [0]），因此可以不要在im中使用诸如for的成语；for im in col可以工作，但不会产生连续的列数据。

事实证明，在处理ufunc时，numpy足够聪明，可以确定哪个索引是内存中变化最快的索引，并将其用于最内层的循环。因此，对于ufunc，在大多数情况下，这两种方法都没有很大的固有优势。另一方面，将.flat与FORTRAN有序数组一起使用将导致非最佳内存访问，因为在扁平化数组中的相邻元素（实际上是迭代器）在内存中不连续。

确实，事实是Python在列表和其他序列上建立索引自然会导致从外到内的排序（第一个索引获得最大的分组，第二个索引获得最大的分组，最后一个索引获得最小的元素）。由于图像数据通常按行存储，因此这对应于最后索引的行在行中的位置。

如果您确实想使用Fortran排序，请意识到有两种方法可以考虑：1）接受第一个索引并不是内存中变化最快的索引，并让所有I / O例程在从内存到磁盘的过程中对数据进行重新排序反之亦然，或者使用numpy的机制将第一个索引映射到变化最快的数据。如果可能，我们建议使用前者。后者的缺点是，除非小心使用'order'关键字，否则许多numpy函数都会产生没有Fortran排序的数组。这样做非常不方便。

否则，我们建议您简单地学习在访问数组元素时反转索引的通常顺序。诚然，它违背了原则，但更符合Python语义和数据的自然顺序。