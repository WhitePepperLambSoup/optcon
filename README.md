# optcon

Dimensional, contract-checked physical computation for optics and machine
learning.

optcon makes four things that optics code normally leaves to memory into
properties the computer can check:

* **dimensions** - `q(1064, "nm") + q(300, "mm")` is a length; adding a tilt
  in mrad to a length is an error, not a number;
* **amplitude vs power** - `T` and `sqrt(T)` are different objects, so the
  square-root class of mistake cannot pass silently;
* **physical invariants** - losslessness, passivity and reciprocity are
  declared and checked, and a waveplate, a polariser and a rotator each fail a
  different one;
* **derivatives** - a gradient is verified against finite differences and its
  adjoint with the `<Jv, w> == <v, J^T w>` dot-product test.

On top of that it carries a closed-form optics toolkit (ABCD matrices,
Gaussian beams, Fresnel, Jones calculus, cavities, fibre, gratings, thin-film
design, diffraction, beam quality, detector noise) and a layer that brings
external solvers under the same type discipline.

It is **not** a replacement for an FDTD, FEM or RCWA solver; it is the
semantic layer above them. See [docs/DESIGN.md](docs/DESIGN.md) for the full
rationale and for the two upstream discrepancies this approach has already
found.

[docs/API.md](docs/API.md) is a generated index of all public functions.

## Install

```bash
pip install -e ".[dev]"        # from the optcon directory
python -m pytest tests -q
```

The closed-form modules need only NumPy and SciPy. The engine adapters look
for their sources under the workspace root, overridable with
`OPTCON_CORPUS_ROOT`; engines that are absent make their tests skip rather
than fail.

## Quick start

```python
from optcon import q, sqrt, power_ratio, amplitude_ratio
from optcon.elements import compose, free_space, thin_lens
from optcon.gaussian import q_from_waist, beam_radius, propagate

wavelength = q(1064.0, "nm")
beam = q_from_waist(q(0.5, "mm"), wavelength)
system = compose(free_space(q(200.0, "mm")), thin_lens(q(100.0, "mm")))
focused = propagate(beam, system)

print(beam_radius(focused, wavelength).to_value("um"))

sqrt(power_ratio(0.81))          # -> 0.9, a field amplitude
sqrt(amplitude_ratio(0.9))       # -> AmplitudeOrderError: that is a sqrt too far
```

## Modules

| Module | Contents |
| --- | --- |
| `units` | dimensional algebra with angle as its own base dimension; unit-expression parser |
| `quantity` | `Quantity`: arithmetic, comparisons, NumPy ufuncs, amplitude order |
| `checks` | unitary / passive / reciprocal contracts; gradient and adjoint verification |
| `specs` | `@checked` signatures and unit inference from dataclass field names |
| `elements` | ABCD matrices, stability, effective focal length |
| `gaussian` | q parameter, beam radius, curvature, Gouy phase, mode overlap, cavity eigenmode |
| `fresnel` | Snell, critical and Brewster angles, reflectance and transmittance |
| `polarization` | Jones states and elements, Stokes parameters, extinction ratio |
| `cavity` | free spectral range, finesse, linewidth, stability, mode sizes |
| `fiber` | numerical aperture, V number, cutoff, mode field, propagation constant |
| `grating` | grating equation, dispersion, Littrow blaze angle, resolving power |
| `thinfilm` | anti-reflection design, quarter-wave stacks, Bragg mirrors |
| `diffraction` | Airy pattern, Rayleigh resolution, encircled energy, Strehl ratio |
| `beam_quality` | M squared and the beam parameter product |
| `noise` | photon energy, responsivity, shot noise, SNR, noise equivalent power |
| `laser` | threshold gain, cavity photon lifetime, slope efficiency, relaxation oscillations |
| `interferometry` | fringe visibility, coherence length, Fabry-Perot etalon |
| `aberrations` | Zernike modes (Noll indexed), Seidel decomposition, wavefront RMS |
| `radiometry` | Planck's law, Stefan-Boltzmann, Wien, etendue, photometry |
| `propagation` | a sampled `Field` and FFT propagation (angular spectrum or Fresnel) |
| `modes` | Gauss-Hermite and Laguerre-Gauss bases, modal decomposition and reconstruction |
| `mtf` | modulation transfer function, closed form and from any PSF |
| `waveguide` | guided TE modes of a symmetric slab, confinement, cutoff |
| `paraxial` | focal length of a singlet, computed by several engines |
| `thermal` | thermal lens focal length, Gaussian aperture and clipping losses |
| `nonlinear` | SHG phase matching, coherence length, quasi-phase-matching period |
| `vector_fields` | two-component fields: Stokes maps, analyzers, radial polarisation |
| `mueller` | Stokes vectors and Mueller matrices, including depolarisation |
| `engines` | registry and adapters for external solvers, plus differential testing |

## What the differential tests found

| Finding | Evidence |
| --- | --- |
| `PyMieScatt.MieQ` defaults to `nMedium=1.00027316` (air) and scales the index silently | shifts `Qext` by 1.5e-3; `nMedium=1.0` agrees with an independent Mie series to 6.4e-14 |
| LightPipes' `Fresnel` convolution propagator is 2-7% wide on a Gaussian beam | `Forvard` matches the closed form to 2.1e-6; the error does not shrink with sampling |
| Two thin-film engines with different internal units agree | `tmm_core` (nm) vs `tmm_fast` (SI metres): 4.1e-08 |
| A ray tracer and an ABCD chain agree on a thick lens | optiland's paraxial `f2` vs `optcon.elements`: 1e-06 |
| Two libraries agree on what "unitary" means | neuroptica's `is_unitary` vs `optcon.checks.is_unitary`: identical on 8 matrices |
| An external adjoint solver passes optcon's gradient verifier | ceviche's autograd `jacobian` vs finite differences: 1e-03 |
| Fresnel interface coefficients agree | `optcon.fresnel` vs `tmm_core`: 1e-12 |
| A wavefront-propagation package reproduces the Airy radius | `poppy` vs `optcon.diffraction`: 2% |
| A waveguide mode solver satisfies its own dispersion relation | every slab mode has residual < 1e-6 and `u^2 + w^2 = V^2` |
| Three independent propagators agree | optcon's FFT, `LightPipes` spectral and `diffractio` CZT: 2e-02 |

Five runnable experiments reproduce all of this (or run all in one pass):

```bash
python -m optcon.benchmarks.run_all               # reproduce all experiments & benchmarks
python -m optcon.examples.design_a_laser          # end-to-end workflow
python -m optcon.examples.experiment_01_guard_bench
python -m optcon.examples.experiment_02_cross_engine
python -m optcon.examples.experiment_03_mie_adjudication
python -m optcon.examples.experiment_04_beam_adjudication
python -m optcon.examples.experiment_05_cavity_thermal_tolerance
```

Experiment 05 writes its publication figure to `docs/figures` when run from the
source checkout. For an installed wheel, it writes to
`optcon-artifacts/figures` in the current working directory so it never needs
to modify `site-packages`. Choose another location with:

```bash
python -m optcon.examples.experiment_05_cavity_thermal_tolerance --output-dir artifacts
```
`design_a_laser` is the one to read first: it designs a two-mirror Nd:YAG
cavity using twelve modules together, and every number in its output is either
a closed-form relation or a contract that either holds or raises.

For modal work, name the basis explicitly. Hermite-Gauss keeps the separable,
low-memory path; Laguerre-Gauss supports radial index `p` and signed azimuthal
charge `ell` (including vortex modes):

```python
from optcon.modes import decompose, laguerre_gauss, reconstruct

mode = laguerre_gauss(x, y, p=1, ell=-1, waist=q(50.0, "um"))
coefficients = decompose(field, waist=q(50.0, "um"), max_order=2, basis="laguerre")
rebuilt = reconstruct(coefficients, field, waist=q(50.0, "um"), basis="laguerre")
```

## Performance

```bash
python -m optcon.benchmarks.bench
```

The benchmark reports wall time and peak allocation per operation. The modal
path was rewritten around the separability of the Hermite-Gauss basis and the
propagation transfer functions around separable phase factors and fewer
temporaries; [docs/PERFORMANCE.md](docs/PERFORMANCE.md) has the before/after
numbers, the two correctness traps the optimisation walked into, and the
sampling regime of each propagator.

| operation (512^2) | speed | peak memory |
| --- | --- | --- |
| modal decomposition | 460x faster | 16.0 -> 0.1 MiB |
| modal reconstruction | ~45x faster | 16 -> 4.1 MiB |
| Fresnel propagation | 3.7x faster | 18.0 -> 12.0 MiB |
| angular-spectrum propagation | 1.9x faster | 24.4 -> 18.4 MiB |
| MTF from a PSF | 1.9x faster | unchanged |
| field construction | 4.5x faster | 10.0 -> 6.0 MiB |

---

以下为中文详细说明。

这个目录是完全独立的实验区。它**不导入、不修改**仓库里任何主程序代码
（`laser_graph_lab/`、`laser_sim_web/` 等都没有被动过），用例全部自带。

## 要解决的问题

在"光学 × 机器学习"的可微仿真里，有几类错误是当前工具链看不见的：

1. **量纲混用**：`wavelength_nm` 和 `L1_mm` 都是裸 `float`，谁都能加谁。
2. **功率与场振幅混用**：`T = 0.81` 和 `t = 0.9` 长得都像"一个小于 1 的数"，
   但 `T = |t|²`，混用会静默给出差一个平方根的结果。
3. **物理不变量无人检查**：无源、无损耗、互易这些性质只存在于作者脑子里。
4. **导数可能不是导数**：反向传播看起来在工作，但梯度和前向不构成伴随对。

## 三个刻意违反 SI 习惯的设计决策

1. **角度是独立量纲**。SI 认为弧度无量纲，于是 `mrad` 可以和裸比值相加。
   在光路对准代码里这几乎总是 bug，所以本库把 `angle` 单独跟踪。
2. **`amp_order` 跟踪"这是场还是功率"**：

   | 取值 | 含义 |
   | --- | --- |
   | `None` | 未跟踪（默认，完全宽松，普通数值代码照常工作） |
   | `0` | 纯比值：算子矩阵、计数 |
   | `1` | 场振幅，例如 `sqrt(T)` |
   | `2` | 功率量，例如 `T`、`|t|²` |

   规则：乘法加指数、除法减指数、`sqrt` 要求偶数阶、加法与比较要求阶相同。
   裸 Python 数字视为未跟踪：加法里继承另一个操作数的单位
   （`q(2,"mm") + 3 == 5 mm`），乘法里视为无量纲（`q(2,"mm") * 3 == 6 mm`）。
3. **单位靠解析而不是枚举**：`unit("mW/cm^2")` 通过前缀 + 乘除 + 整数幂组合得到。

## 模块

| 文件 | 职责 |
| --- | --- |
| `errors.py` | 错误分级：量纲错 / 振幅阶错 / 契约违背 / 伴随检验失败 |
| `units.py` | `Dimension`、`Unit`、单位表达式解析与换算 |
| `quantity.py` | `Quantity`：算术、比较、numpy ufunc 集成、振幅阶代数 |
| `checks.py` | 不变量契约（酉性/无源性/互易性）与导数、伴随检验 |
| `specs.py` | 采用层：`@checked` 装饰器 + 从 dataclass 字段名推导单位 |
| `engines/` | 把工作区里已有的光学引擎接入类型层（见下） |
| `examples/` | 自带的小型实验，不依赖主程序 |

## 引擎整合层（engines/）

工作区里已经调研过 40 个光学库，其中 8 个在当前环境可以导入。这一层把它们
**接进类型系统**，而不是重写它们：

| 引擎 | 领域 | 长度单位 | 角度单位 | 备注 |
| --- | --- | --- | --- | --- |
| `tmm_core` | 薄膜 | nm | rad | Byrnes 参考实现，标量，同时返回 r,t 振幅与 R,T 功率 |
| `tmm_fast` | 薄膜 | **m（SI）** | rad | 可微/向量化，只吃批量数组 |
| `miepython` | Mie 散射 | nm | rad | 签名 `(m, 直径, 波长)`；第三个返回值是 **qback 不是 qabs** |
| `PyMieScatt` | Mie 散射 | nm | rad | 签名 `(m, 波长, 直径)`；参数量纲顺序与上者相反 |
| `optcon_reference` | Mie 散射 | nm | rad | 本库自写的 Mie 级数（SciPy Bessel/Hankel），用于裁决上面两者 |
| `optcon_beam_reference` | 光束传播 | mm | rad | 高斯光束解析解 `w(z)=w₀√(1+(z/z_R)²)` |
| `lightpipes_forvard` | 光束传播 | mm | rad | LightPipes 谱方法传播器 |
| `lightpipes_fresnel` | 光束传播 | mm | rad | LightPipes 卷积传播器；**同一库、不同传播器、精度不同** |
| `diffractio` | 衍射 | **um** | rad | 自带单位常量（`um=1.0, nm=0.001, mm=1000.0`），内部基准是微米；适配器待做 |
| `tracepy` | 光线追迹 | mm | rad | Spencer-Murty 顺序追迹；适配器待做 |
| `pyoptools` | 光线追迹 | mm | rad | Cython 追迹与波前；适配器待做 |

这些约定全部作为**声明式数据**存在注册表里，适配器在边界转换单位。于是：

* 同一个物理问题传 `nm` 或 `um` 结果完全一致；
* 30 deg 和 30 mrad 会被区分开（而不是都当成裸数）；
* `T` 返回的是**功率比**（`amp_order=2`），不会被误当成场振幅；
* 引擎缺失时注册表会报告原因，而不是让调用方撞一个陌生异常。

### 新增能力：跨引擎差分验证

`compare_across_engines()` 把同一个物理查询同时跑在多个独立实现上，
在统一单位下比较，并按声明的容差给出裁决。这是这类库原来都没有的东西。

## 运行
 
```bash
# 测试（从仓库根目录）
python -m pytest tests -q

# 一键复现所有实验与基准
python -m optcon.benchmarks.run_all

# 实验 01：注入式 bug 检测率 / 误报率
python -m optcon.examples.experiment_01_guard_bench

# 实验 02：跨引擎差分验证
python -m optcon.examples.experiment_02_cross_engine

# 实验 03：Mie 三方裁决
python -m optcon.examples.experiment_03_mie_adjudication

# 实验 04：光束传播 vs 解析解
python -m optcon.examples.experiment_04_beam_adjudication
python -m optcon.examples.experiment_05_cavity_thermal_tolerance
```

Experiment 05 从源码仓库运行时会把论文图写入 `docs/figures`。安装 wheel
后，默认写入当前工作目录下的 `optcon-artifacts/figures`，不会修改
`site-packages`。也可以显式指定输出目录：

```bash
python -m optcon.examples.experiment_05_cavity_thermal_tolerance --output-dir artifacts
```
## 实验 01 的结果（当前）

16 个用例：9 个故意注入的物理错误 + 7 个合法操作。

| 指标 | 结果 |
| --- | --- |
| 注入 bug 检出 | 9/9 |
| 合法代码误报 | 0/7 |
| 静默数值分歧 | 2/5（float 路径给出不同且错误的数字） |

两条静默分歧是最有价值的部分，因为它们**不抛异常**：

* `sin(0.3)` 而 0.3 的实际含义是 0.3 mrad → float 路径得 0.29552，
  物理正确的是 0.0003（差 1000 倍）。
* 把 1064 nm 直接加进 mm 的代数里 → float 路径得 1364，
  正确结果是 300.001 mm。

## 实验 02 的结果（跨引擎，当前）

同一个物理查询分别跑在两套独立实现上，容差声明为 rtol = 1e-6：

| 领域 | 引擎对 | 最大相对分歧 | 裁决 |
| --- | --- | --- | --- |
| 薄膜 | `tmm_core` vs `tmm_fast` | **4.1e-08** | 一致 |
| Mie | `miepython` vs `PyMieScatt` | **1.9e-03** | 不一致 |

薄膜两套实现跨 3 个波长 × 2 个角度都一致到 1e-8 量级；
而 Mie 两套实现在尺寸参数 x = 0.29 到 28.6 的全区间内**稳定分歧约 0.2%**，
且最差指标在 `Qsca` 和 `Qabs` 之间跳动。

这说明两件事：一是"独立实现"是计算光学最便宜的基准来源，
二是**分歧不需要有 bug**——它只是没人量过。声明容差之后，
这条从"未知"变成了可检查的断言。

## 实验 03 的结果（三方裁决：到底谁错了？）

两个库分歧本身不构成证据，所以补了第三个实现（本库自写的 Mie 级数）来裁决：

| 实现 | 与独立参考的最大偏差（Qext） |
| --- | --- |
| `miepython` | **1.8e-10** |
| `PyMieScatt` | **1.5e-03** |

而且排除了最可能的借口：把项数从 Wiscombe 的 `ceil` 换成 `round`、再加 8 项，
参考实现的结果**十位有效数字完全不变**——说明它已经收敛，分歧不是级数截断。
偏差在 x ≈ 1–30 区间最大，也就是 Mie 曲线结构最丰富的区域。

## 实验 04 的结果（成熟库 vs 解析解）

LightPipes 提供两个传播器。对高斯光束有精确解 `w(z)=w₀√(1+(z/z_R)²)`，
不需要第三个库来当裁判：

| 传播器 | 与解析解的最大偏差 | 裁决 |
| --- | --- | --- |
| `Forvard`（谱方法） | **2.1e-06** | 一致 |
| `Fresnel`（卷积） | **6.9e-02** | 不一致 |

卷积传播器在整个 z 区间偏宽 2%~7%，且**加密采样（256→4096）后误差不降**，
所以不是分辨率问题。同一个成熟库的两个命令，没有任何一个会警告你。

## 新增能力：CI 门禁

`assert_engines_agree()` 是差分比较的"断言"形式：独立实现是基准，
而一个从不允许失败的基准不是基准。

## 尚未实现（路线图）

* `diffractio`、`tracepy`、`pyoptools` 的适配器：单位约定已登记，
  但还没有统一的被测量可以做差分比较（衍射需要网格对齐，追迹需要面型约定）。
* 接不进来的引擎（`optiland`、`neuroptica` 缺 numba；`ceviche` 缺 autograd；
  `torchoptics`、`rayoptics` 是布局问题）已在注册表里记录单位约定，
  等依赖具备即可启用。
* 场表示层：离散化/基函数作为类型（模态基 vs 网格场不可混算）。
* 效应层：随机参数与硬件副作用作为显式标注。
* 面向 agent 的类型化修复语言，替代手搓的 if-else DSL。
* torch/jax 后端：当前数值载体是 numpy。

已完成：打包（`pyproject.toml`）、CI 工作流、LICENSE、CONTRIBUTING、
CHANGELOG、设计文档（`docs/DESIGN.md`）、`ruff` 与 `mypy` 全绿。

## 需要注意的边界

这个库只能检查**你已经写下来的约束**。它抓不住"语义标错"——
例如给一个波长变量起名 `L1_mm`，量纲检查无从判断。奖励设计、数据稀缺、
仿真到现实的差距，都不在类型系统的能力范围内。
