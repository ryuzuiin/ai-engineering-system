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
