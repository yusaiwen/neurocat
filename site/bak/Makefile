# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
BUILDDIR      = build

# 定义两个源文件的路径
SOURCEDIR_EN  = source_en
SOURCEDIR_ZH  = source_zh

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR_EN)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "自定义构建命令:"
	@echo "  html_en     生成英文版文档 (build/html/en)"
	@echo "  html_zh     生成中文版文档 (build/html/zh)"
	@echo "  html_all    同时生成双语文档，并生成跳转首页"
	@echo "  live_en     实时预览英文版 (需安装 sphinx-autobuild)"
	@echo "  live_zh     实时预览中文版 (需安装 sphinx-autobuild)"

.PHONY: help Makefile html_en html_zh html_all live_en live_zh

# 1. 构建英文版
html_en:
	@echo "正在构建英文版..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR_EN)" "$(BUILDDIR)/html/en" $(SPHINXOPTS)
	@echo "英文版构建完成: $(BUILDDIR)/html/en/index.html"

# 2. 构建中文版
html_zh:
	@echo "正在构建中文版..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR_ZH)" "$(BUILDDIR)/html/zh" $(SPHINXOPTS)
	@echo "中文版构建完成: $(BUILDDIR)/html/zh/index.html"

# 3. 同时构建两个版本，并生成一个根目录的重定向文件
html_all: html_en html_zh
	@echo "正在生成根目录跳转文件..."
	@echo '<meta http-equiv="refresh" content="0; url=./en/index.html" />' > "$(BUILDDIR)/html/index.html"
	@echo "全部完成!"

# 4. 实时预览英文版 (端口 8000)
live_en:
	sphinx-autobuild "$(SOURCEDIR_EN)" "$(BUILDDIR)/html/en" --port 8000 --host 0.0.0.0

# 5. 实时预览中文版 (端口 8001 - 防止冲突)
live_zh:
	sphinx-autobuild "$(SOURCEDIR_ZH)" "$(BUILDDIR)/html/zh" --port 8001 --host 0.0.0.0

# 清理命令
clean:
	rm -rf $(BUILDDIR)/*

# 默认的捕获规则 (如果有其他命令，默认对英文版操作)
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR_EN)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
