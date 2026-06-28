# Know-Surgery Paper

Know-Surgery: A Unified Toolkit for Controllable Knowledge Update in Large Language Models (ACL 2026 Demo Track)

## 本次更新内容

对照 open-unlearning 最新代码，完成了以下工作：

1. **Plan 3.1 架构决策** — 更新方法统计：48 种方法（14 unlearn + 15 edit + 6 inject + 7 MM unlearn + 6 MM edit），42 个模型配置，17+ 数据集 handler，12+ 评估器
2. **Plan 3.3 项目结构** — 对照 open-unlearning 实际目录结构重写，包含 src/、configs/、new_ui/、saves/ 等完整路径
3. **论文各章节更新** — 01-introduction、02-system、03-features、04-demo、05-evaluation、06-conclusion、appendix 全部对照最新代码更新方法数量、分类、数据集列表、模型列表
4. **LaTeX 编译通过** — 使用 tectonic 编译，无 error，PDF 已生成

**未完成**：3.2 实验部分（evaluation 表格中数据为 `--` 占位符，待后续跑实验填充）

## 论文结构

```
paper/
├── main.tex                 # 主文件
├── sections/
│   ├── 01-introduction.tex  # 引言
│   ├── 02-system.tex        # 系统架构
│   ├── 03-features.tex      # 核心特性
│   ├── 04-demo.tex          # 演示
│   ├── 05-evaluation.tex    # 评估 (待填数据)
│   ├── 06-conclusion.tex    # 结论
│   └── appendix.tex         # 附录 (完整方法/数据集/模型列表)
├── references.bib           # 参考文献
├── acl.sty                  # ACL 样式文件
├── acl_natbib.bst           # 参考文献样式
└── main.pdf                 # 编译产物
```

## 编译 LaTeX

### 方法一：tectonic（推荐，已安装）

```bash
cd /home/liumingxuan/paper
tectonic main.tex
# 输出: main.pdf
```

### 方法二：latexmk（需安装 texlive）

```bash
# 安装
sudo apt install texlive-full latexmk

# 编译
cd /home/liumingxuan/paper
latexmk -pdf -interaction=nonstopmode main.tex

# 清理中间文件
latexmk -c
```

### 方法三：手动 pdflatex

```bash
cd /home/liumingxuan/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
