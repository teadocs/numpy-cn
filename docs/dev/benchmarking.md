---
meta:
  - name: keywords
    content: NumPy 基准测试
  - name: description
    content: 用 ​​irspeed Velocity 对 NumPy 进行基准测试...
---

# NumPy 基准测试

用 ​​irspeed Velocity 对 NumPy 进行基准测试。

## 用法

除非另有说明，否则 A​​irspeed Velocity 会自行管理构建和 Python virtualenvs。
一些基准测试功能 ``runtests.py`` 也告诉ASV使用编译的 NumPy ``runtests.py``。
要运行基准测试，您无需在当前的Python环境中安装NumPy的开发版本。

针对当前签出的NumPy版本运行基准测试（不记录结果）：

``` python
python runtests.py --bench bench_core
```

比较基准测试结果与其他版本的变化：

``` python
python runtests.py --bench-compare v1.6.2 bench_core
```

运行ASV命令（记录结果并生成HTML）：

``` python
cd benchmarks
asv run --skip-existing-commits --steps 10 ALL
asv publish
asv preview
```

有关如何使用的更多信息``asv``可以在[ASV文档中](https://asv.readthedocs.io/)找到
命令行帮助可以像往常一样通过 ``asv --help`` 和 ``asv run --help`` 获取。

## 编写基准测试

有关如何编写基准测试的基础知识，请参阅[ASV文档](https://asv.readthedocs.io/)。

以下事情需要考虑：

- 基准套件应该可以使用任何NumPy版本导入。
- 基准参数等不应取决于安装的NumPy版本。
- 尽量保持基准测试的运行时间合理。
- 喜欢ASV的 ``time_`` 基准测试时间的方法，而不是通过 ``time.clock`` 编写时间测量，即使在编写基准时需要一些方案。
- 通常应该在 ``setup`` 方法而不是 ``time_`` 方法中放置数组等，以避免计算准备时间和基准操作的时间。
- 请注意，在访问内存之前，使用 ``np.empty`` 或` `np.zeros`` 不在物理内存中分配大型数组。如果这是所需的行为，请务必在设置功能中对其进行注释。如果您正在对算法进行基准测试，则用户不太可能在新创建的 空/零 数组上执行所述算法。可以通过调用``np.ones``或
 ``arr.fill(value)``创建数组后强制在设置阶段抛出页面异常。
