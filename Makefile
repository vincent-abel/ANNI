CC=gcc -g
# `make` with no target builds ./anni (the per-object rules below come first).
.DEFAULT_GOAL := all
# Include roots: net/ (neural network), num/ (numerical engine), lang/ (language model), cli/.
INCLUDES=-I. -Inet -Inum -Ilang -Icli -Icompat
# OPT=-O2 enables auto-vectorization on contiguous ann_dot / mat_mul loops.
OPT ?= -O2
OPENMP ?= -fopenmp
# ARCH may add -mavx2 / -mavx512f when the host supports it.
ARCH ?=
# Ncurses control tower: enabled when headers are present.
NCURSES_CFLAGS=
NCURSES_LIBS=
ifneq ($(wildcard /usr/include/ncurses.h),)
NCURSES_CFLAGS=-DANN_NCURSES
NCURSES_LIBS=-lncurses
else ifneq ($(wildcard /usr/include/ncurses/curses.h),)
NCURSES_CFLAGS=-DANN_NCURSES -I/usr/include/ncurses
NCURSES_LIBS=-lncurses
endif
CFLAGS=-W -Wall -ansi -pedantic -std=c99 $(OPT) $(OPENMP) $(ARCH) $(INCLUDES) $(NCURSES_CFLAGS)
LDFLAGS=-lm $(OPENMP) $(NCURSES_LIBS)
EXEC=anni
OBJDIR=objects

NET_SRC=net/neuron.c net/layer.c net/dataset.c net/optim.c net/serialize.c
NUM_SRC=num/vec.c num/mat.c num/tensor.c num/ops.c num/arena.c num/simd.c num/gpu_stub.c
NVCC ?= nvcc
CUDA_OBJ=$(OBJDIR)/gpu_cuda.o
LANG_SRC=lang/tokenize.c lang/vocab.c lang/bpe.c lang/embed.c lang/posenc.c lang/attn.c lang/ffn.c lang/layernorm.c lang/resid.c lang/block.c lang/lm.c lang/generate.c lang/gradutil.c lang/lm_adam.c lang/dataset_text.c lang/dataset_stream.c lang/lm_serialize.c lang/lm_train.c lang/lm_bench.c lang/lm_info.c lang/quant.c lang/decode.c
CLI_SRC=cli/tower.c cli/chat_session.c cli/chat_ui.c
LIB_SRC=$(NET_SRC) $(NUM_SRC) $(LANG_SRC)
LIB_OBJ=$(patsubst %.c,$(OBJDIR)/%.o,$(notdir $(LIB_SRC)))
CLI_OBJ=$(OBJDIR)/tower.o $(OBJDIR)/chat_session.o $(OBJDIR)/chat_ui.o
MAIN_OBJ=$(OBJDIR)/main.o
OBJ=$(LIB_OBJ) $(CLI_OBJ) $(MAIN_OBJ)
TEST=tests/test_ann
GUI=gui/ann-gui

# Map basename.o -> source path
$(OBJDIR)/neuron.o: net/neuron.c
$(OBJDIR)/layer.o: net/layer.c
$(OBJDIR)/dataset.o: net/dataset.c
$(OBJDIR)/optim.o: net/optim.c
$(OBJDIR)/serialize.o: net/serialize.c
$(OBJDIR)/vec.o: num/vec.c
$(OBJDIR)/mat.o: num/mat.c
$(OBJDIR)/tensor.o: num/tensor.c
$(OBJDIR)/ops.o: num/ops.c
$(OBJDIR)/arena.o: num/arena.c
$(OBJDIR)/simd.o: num/simd.c
$(OBJDIR)/gpu_stub.o: num/gpu_stub.c
$(OBJDIR)/tokenize.o: lang/tokenize.c
$(OBJDIR)/vocab.o: lang/vocab.c
$(OBJDIR)/bpe.o: lang/bpe.c
$(OBJDIR)/embed.o: lang/embed.c
$(OBJDIR)/posenc.o: lang/posenc.c
$(OBJDIR)/attn.o: lang/attn.c
$(OBJDIR)/ffn.o: lang/ffn.c
$(OBJDIR)/layernorm.o: lang/layernorm.c
$(OBJDIR)/resid.o: lang/resid.c
$(OBJDIR)/block.o: lang/block.c
$(OBJDIR)/lm.o: lang/lm.c
$(OBJDIR)/generate.o: lang/generate.c
$(OBJDIR)/gradutil.o: lang/gradutil.c
$(OBJDIR)/lm_adam.o: lang/lm_adam.c
$(OBJDIR)/dataset_text.o: lang/dataset_text.c
$(OBJDIR)/dataset_stream.o: lang/dataset_stream.c
$(OBJDIR)/lm_train.o: lang/lm_train.c
$(OBJDIR)/lm_serialize.o: lang/lm_serialize.c
$(OBJDIR)/lm_bench.o: lang/lm_bench.c
$(OBJDIR)/lm_info.o: lang/lm_info.c
$(OBJDIR)/quant.o: lang/quant.c
$(OBJDIR)/kv.o:
$(OBJDIR)/decode.o: lang/decode.c
$(OBJDIR)/tower.o: cli/tower.c
$(OBJDIR)/chat_session.o: cli/chat_session.c
$(OBJDIR)/chat_ui.o: cli/chat_ui.c
$(OBJDIR)/main.o: main.c

all: $(EXEC) ann

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: | $(OBJDIR)
	$(CC) -o $@ -c $(firstword $(filter %.c,$^)) $(CFLAGS)

anni: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

# Compat symlink so scripts using ./ann keep working.
ann: anni
	ln -sfn anni ann

# Unit tests rebuild the library with a small LM so make test stays fast on CPU.
TEST_LM_FLAGS=-DLM_D_MODEL=64 -DLM_N_HEADS=4 -DLM_D_FF=256 -DLM_N_BLOCKS=2
TEST_OBJDIR=$(OBJDIR)/test
TEST_LIB_OBJ=$(patsubst $(OBJDIR)/%,$(TEST_OBJDIR)/%,$(LIB_OBJ))
vpath %.c net num lang

$(TEST_OBJDIR):
	mkdir -p $(TEST_OBJDIR)

$(TEST_OBJDIR)/%.o: %.c | $(TEST_OBJDIR)
	$(CC) -o $@ -c $< $(CFLAGS) $(TEST_LM_FLAGS)

$(TEST): tests/test_ann.c $(TEST_LIB_OBJ)
	$(CC) $(CFLAGS) $(TEST_LM_FLAGS) -o $@ tests/test_ann.c $(TEST_LIB_OBJ) $(LDFLAGS)

test: $(TEST)
	./$(TEST)

gui: $(GUI)

$(GUI): gui/bridge.c $(LIB_SRC) main.h
	$(CC) $(CFLAGS) -o $@ gui/bridge.c $(LIB_SRC) $(LDFLAGS)

run-gui: gui
	python3 gui/server.py

# Real corpus (never vendored): Simple English Wikipedia slice -> data/wiki_simple.txt.
# Override with WIKI_LANG=en MAX_MB=500 WIKI_DUMP=YYYYMMDD; see docs/real-corpora-scale.md.
data-wiki:
	./scripts/fetch_wiki.sh

.PHONY: clean mrproper test gui run-gui all cuda data-wiki autotrain \
	autotrain-cpu cuda-autotrain

# AT1: opinionated ./autotrain — CUDA by default (Target path is GPU-first).
autotrain: cuda-autotrain

# CPU stub only (teaching / no nvcc). Prefer `make autotrain` on a CUDA box.
autotrain-cpu: $(LIB_OBJ) autotrain.c
	$(CC) $(CFLAGS) -o autotrain autotrain.c $(LIB_OBJ) $(LDFLAGS)

cuda-autotrain: $(filter-out $(OBJDIR)/gpu_stub.o,$(LIB_OBJ)) $(CUDA_OBJ) autotrain.c
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "nvcc not found; cannot build CUDA autotrain (try: make autotrain-cpu)"; \
		exit 1; \
	fi
	$(CC) $(CFLAGS) -o autotrain autotrain.c \
	    $(filter-out $(OBJDIR)/gpu_stub.o,$(LIB_OBJ)) $(CUDA_OBJ) \
	    $(LDFLAGS) $(CUDA_LIBS)
	@echo "built CUDA ./autotrain (backend will be reported at runtime)"

# Optional CUDA build: replaces gpu_stub.o with gpu_cuda.o (requires nvcc).
# ANN_CUBLAS=1 (default) links cuBLAS for GEMM; set ANN_CUBLAS=0 for teaching kernels only.
# Prefer CUDA 13.x then 12.8 so Blackwell gets sm_120 + Lt/FP8.
ANN_CUBLAS ?= 1
CUDA_HOME ?= $(firstword \
	$(wildcard $(HOME)/cuda13/usr/local/cuda-13*) \
	$(wildcard $(HOME)/cuda13/usr/local/cuda) \
	$(wildcard $(HOME)/cuda12/usr/local/cuda-12.8) \
	$(wildcard /usr/local/cuda-13*) \
	$(wildcard /usr/local/cuda))
ifneq ($(wildcard $(CUDA_HOME)/bin/nvcc),)
NVCC := $(CUDA_HOME)/bin/nvcc
CUDA_INC := -I$(CUDA_HOME)/include
CUDA_LDFLAGS := -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64
# RTX 50-class (sm_120) when the toolkit lists it; else let nvcc pick.
CUDA_ARCH_FLAG := $(shell $(NVCC) --list-gpu-arch 2>/dev/null | grep -q compute_120 && echo -arch=sm_120)
else
CUDA_INC :=
CUDA_LDFLAGS :=
CUDA_ARCH_FLAG :=
endif
CUDA_NVCC_FLAGS=-O2 -I. -Inum -Ilang -Icompat $(CUDA_INC) $(CUDA_ARCH_FLAG) --use_fast_math
CUDA_LIBS=$(CUDA_LDFLAGS) -lcudart
ifeq ($(ANN_CUBLAS),1)
CUDA_NVCC_FLAGS+=-DANN_CUBLAS
CUDA_LIBS+=-lcublas -lcublasLt
endif

cuda: $(filter-out $(OBJDIR)/gpu_stub.o,$(OBJ)) $(CUDA_OBJ)
	@if ! command -v $(NVCC) >/dev/null 2>&1 && [ ! -x "$(NVCC)" ]; then \
		echo "nvcc not found; keeping CPU stub (make / make test still work)"; \
		exit 1; \
	fi
	$(CC) -o $(EXEC) $(filter-out $(OBJDIR)/gpu_stub.o,$^) $(LDFLAGS) $(CUDA_LIBS) -lstdc++
	ln -sfn anni ann

$(CUDA_OBJ): num/gpu_cuda.cu | $(OBJDIR)
	$(NVCC) $(CUDA_NVCC_FLAGS) -c num/gpu_cuda.cu -o $@

clean:
	rm -rf $(OBJDIR)

mrproper: clean
	rm -rf $(EXEC) ann
	rm -rf $(TEST)
	rm -rf $(GUI)
	rm -rf autotrain
	rm -rf *~
