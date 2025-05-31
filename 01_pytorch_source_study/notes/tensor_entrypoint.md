# 📌 torch.tensor() 的源码入口分析

> 本笔记记录了我对 PyTorch 中 `torch.tensor()` 构造张量过程的精读路径。目标是搞清楚 Python list 是如何一步步变成 Tensor，并最终分配到 CPU/GPU 上的内存中的。

---

## 🚪 入口函数

首先，`torch.tensor(data)` 实际上调用的是：

```python
# torch/__init__.py
def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
    return _tensor(data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)

🧩 _tensor() 的实现

位置：torch/_utils.py

def _tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    if isinstance(data, Tensor):
        if dtype is not None and data.dtype != dtype:
            data = data.to(dtype)
        if device is not None and data.device != device:
            data = data.to(device)
        return data.requires_grad_(requires_grad)
    return torch.as_tensor(data, dtype=dtype, device=device)
📝 我的理解：
如果 data 已经是 Tensor，就只做 dtype/device 的转换。
否则进入下一层 —— torch.as_tensor()。

🔍 torch.as_tensor()

位置：torch/functional.py

def as_tensor(data, dtype=None, device=None):
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device) if (dtype is not None or device is not None) else data
    return tensor(data, dtype=dtype, device=device)
注意这里：

这个 tensor() 并不是最外层的 torch.tensor()，而是 Python bindings 到 C++ 的接口。
🧵 到这里为止我学到了：

PyTorch 对输入数据做了多层封装，确保数据在传入时被安全转换成 tensor。
真正的 Tensor 构造函数在 C++ backend 里，之后我要进一步研究 torch/csrc/utils/tensor_new.cpp，分析：
Python list → contiguous buffer 的构建过程
Tensor 的 shape 推断逻辑
数据类型（dtype）和 device 的默认处理流程


User → torch.tensor()
         ↓
   _tensor()  ← torch/_utils.py
         ↓
   as_tensor()  ← torch/functional.py
         ↓
torch._C._VariableFunctionsClass.tensor  ← pybind11 绑定
         ↓
C++ 实现（aten::tensor / from_blob / frombuffer / copy_ 等）

torch/
├── __init__.py            ← 你熟悉的导入文件
├── _utils.py              ← _tensor() 定义在这
├── functional.py          ← as_tensor() 在这里
└── _C/                    ← C++ bindings 的核心模块（用 Pybind11 构建）


对tensor的调用链是这样的：
torch.tensor()  ← 绑定到 _C 中的 built-in 函数
         ↓
_C._TensorBase 或 pybind11 C++ wrapper
         ↓
内部根据输入类型决定是 new Tensor, clone, convert...
         ↓
调用 C++ 实现：aten::tensor(), from_blob(), empty(), copy_(), 等等


python的壳子是_tensor.py和_utils.py
这两个是围绕tensor（）行为的用户侧包装与默认行为配置逻辑，但不是核心tensor创建函数。
之前追踪的_tensor()和as_tensor()属于前处理模块，真正的构造，是通过C++调用完成的。

torch.tensor()构造路径总览图

┌─────────────────────────────┐
│        torch.tensor()       │  ← 你在 notebook 中调用
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│     torch._C._VariableFunctionsClass.tensor     │  
│     ← 动态绑定的 Python 接口（用 C++ 实现）        │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│   Pybind11: torch/csrc/autograd/generated/      │
│   python_torch_functions.cpp + DispatchStub     │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│   C++ Backend: aten::tensor() / from_blob()     │
│     ← 真正构造 Tensor 的 C++ 实现逻辑            │
└─────────────────────────────┘

| 模块                    | 用途                                                      | 是否建议学习              |
| --------------------- | ------------------------------------------------------- | ------------------- |
| `torch/_utils.py`     | `_tensor()` 函数定义。是 `tensor()` 的包装器之一。                   | ✅ 必学                |
| `torch/functional.py` | 包含 `as_tensor()` 和参数预处理逻辑。                              | ✅ 建议读               |
| `torch/_C/`           | 用于导入 C++ 接口（如 `_C._VariableFunctionsClass.tensor`）      | ✅ 了解即可              |
| `aten/`               | C++ 中真正构造 Tensor 的位置。比如 `aten::CPU`, `aten::from_blob`。 | ✅ 深入理解 Tensor 机制时阅读 |


在 torch._utils._tensor() 中用 deepcopy 干嘛？
用户传来的结构（如list，dict，np。ndarray）不被副作用污染
某些中间变量的状态要完整复制出来，防止原始对象被修改

⚠️ 注意事项：
1. deepcopy（）对于大数组或复杂嵌套结构非常慢→ 所以 PyTorch 只在必要时用它
2. 在你阅读 PyTorch 源码时，看到 deepcopy() 往往说明：
“这个结构不能动原来的副本，要完全隔离处理”