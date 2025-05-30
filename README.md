# AI Engineering System｜AIエンジニアリング体系構築記録

🎯 This repository documents my structured journey to transition from a data science background to a full-stack AI Engineer. It focuses on **bottom-up mastery**—from PyTorch internals to LLM reconstruction and time series foundation model adaptation.

本仓库记录了我从数据科学者向系统型 AI 工程师进化的全过程：  
从 PyTorch 源码的深入理解，到 NanoGPT 的模型重构，再到 TimesFM 的时间序列任务改写，逐步构建我的技术信用资产与系统构建能力。

---

## 📁 项目结构｜Project Structure




---

## 💡 核心目标｜Core Objectives

- 🎯 **底层掌握力**：精读 PyTorch，掌握 Tensor 创建与模块调用的核心机制
- 🧠 **模型重构能力**：改写 NanoGPT，搭建与训练完整 Transformer
- 📈 **任务迁移力**：适配并重构 TimesFM，使其支持多类型时序任务
- 📓 **结构性表达**：构建工程 + 学术 + 项目三位一体的知识资产系统

---

## 🔍 子项目一览｜Subproject Overview

### 1. [PyTorch Source Study](./01_pytorch_source_study/)
- Entrypoint analysis: `torch.tensor()` → `_tensor()` → `as_tensor()` → C++ backend
- Dispatch system, memory layout, Tensor storage
- Module system exploration: `nn.Module`, hooks, forward/backward mechanics

### 2. [NanoGPT Rebuild](./02_nanogpt_rebuild/)
- From Shakespeare to Japanese corpus
- Full rewrite of minimal GPT structure with comments and visualizations
- Loss behavior tracking, learning curve plotting

### 3. [TimesFM Rewrite](./03_timesfm_rewrite/)
- Slimmed and task-adapted TimesFM
- Customized training for small datasets
- Integrated sliding window, forecasting task wrappers

---

## 🧭 我的方向｜My Trajectory

我希望将这些工程练习发展为：

- 📚 一份持续演进的 AI 技术信用资产  
- 🏗️ 一套系统构造力的公开可见化作品  
- 💼 面向 AI Engineer / Research Engineer  的职业桥梁

---

Feel free to explore any subfolder — each contains markdown notebooks, source code, and visual maps of learning.
