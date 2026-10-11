CC=gcc -g
# Include roots: net/ (neural network), num/ (numerical engine), lang/ (language model), cli/.
INCLUDES=-I. -Inet -Inum -Ilang -Icli
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
EXEC=ann
OBJDIR=objects

NET_SRC=net/neuron.c net/layer.c net/dataset.c net/optim.c net/serialize.c
NUM_SRC=num/vec.c num/mat.c num/tensor.c num/ops.c num/arena.c num/simd.c num/gpu_stub.c
NVCC ?= nvcc
CUDA_OBJ=$(OBJDIR)/gpu_cuda.o
LANG_SRC=lang/tokenize.c lang/vocab.c lang/bpe.c lang/embed.c lang/posenc.c lang/attn.c lang/ffn.c lang/layernorm.c lang/resid.c lang/block.c lang/lm.c lang/generate.c lang/gradutil.c lang/lm_adam.c lang/dataset_text.c lang/dataset_stream.c lang/lm_serialize.c lang/lm_train.c lang/lm_bench.c lang/lm_info.c lang/quant.c lang/kv.c
CLI_SRC=cli/tower.c
LIB_SRC=$(NET_SRC) $(NUM_SRC) $(LANG_SRC)
LIB_OBJ=$(patsubst %.c,$(OBJDIR)/%.o,$(notdir $(LIB_SRC)))
CLI_OBJ=$(OBJDIR)/tower.o
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
$(OBJDIR)/kv.o: lang/kv.c
$(OBJDIR)/tower.o: cli/tower.c
$(OBJDIR)/main.o: main.c

all: $(EXEC)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: | $(OBJDIR)
	$(CC) -o $@ -c $(firstword $(filter %.c,$^)) $(CFLAGS)

ann: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

$(TEST): tests/test_ann.c $(LIB_OBJ)
	$(CC) $(CFLAGS) -o $@ tests/test_ann.c $(LIB_OBJ) $(LDFLAGS)

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

.PHONY: clean mrproper test gui run-gui all cuda data-wiki autotrain

# AT1: opinionated ./autotrain beside ./ann (same libs).
autotrain: $(LIB_OBJ) autotrain.c
	$(CC) $(CFLAGS) -o autotrain autotrain.c $(LIB_OBJ) $(LDFLAGS)

cuda-autotrain: $(filter-out $(OBJDIR)/gpu_stub.o,$(LIB_OBJ)) $(CUDA_OBJ) autotrain.c
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "nvcc not found"; exit 1; \
	fi
	$(CC) $(CFLAGS) -o autotrain autotrain.c \
	    $(filter-out $(OBJDIR)/gpu_stub.o,$(LIB_OBJ)) $(CUDA_OBJ) \
	    $(LDFLAGS) $(CUDA_LIBS)

# Optional CUDA build: replaces gpu_stub.o with gpu_cuda.o (requires nvcc).
# ANN_CUBLAS=1 (default) links cuBLAS for GEMM; set ANN_CUBLAS=0 for teaching kernels only.
ANN_CUBLAS ?= 1
CUDA_NVCC_FLAGS=-O2 -I. -Inum -Ilang
CUDA_LIBS=-lcudart
ifeq ($(ANN_CUBLAS),1)
CUDA_NVCC_FLAGS+=-DANN_CUBLAS
CUDA_LIBS+=-lcublas
endif

cuda: $(filter-out $(OBJDIR)/gpu_stub.o,$(OBJ)) $(CUDA_OBJ)
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "nvcc not found; keeping CPU stub (make / make test still work)"; \
		exit 1; \
	fi
	$(CC) -o $(EXEC) $(filter-out $(OBJDIR)/gpu_stub.o,$^) $(LDFLAGS) $(CUDA_LIBS)

$(CUDA_OBJ): num/gpu_cuda.cu | $(OBJDIR)
	$(NVCC) $(CUDA_NVCC_FLAGS) -c num/gpu_cuda.cu -o $@

clean:
	rm -rf $(OBJDIR)

mrproper: clean
	rm -rf $(EXEC)
	rm -rf $(TEST)
	rm -rf $(GUI)
	rm -rf autotrain
	rm -rf *~
