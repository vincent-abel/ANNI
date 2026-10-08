CC=gcc -g
# Include roots: net/ (neural network), num/ (numerical engine), lang/ (language model).
INCLUDES=-I. -Inet -Inum -Ilang
CFLAGS=-W -Wall -ansi -pedantic -std=c99 $(INCLUDES)
LDFLAGS=-lm
EXEC=ann
OBJDIR=objects

NET_SRC=net/neuron.c net/layer.c net/dataset.c net/optim.c net/serialize.c
NUM_SRC=num/vec.c num/mat.c num/tensor.c num/ops.c
LANG_SRC=lang/tokenize.c lang/vocab.c lang/embed.c lang/posenc.c lang/attn.c lang/ffn.c lang/layernorm.c lang/resid.c lang/block.c lang/lm.c lang/generate.c lang/gradutil.c lang/lm_adam.c lang/dataset_text.c lang/lm_serialize.c lang/lm_train.c
LIB_SRC=$(NET_SRC) $(NUM_SRC) $(LANG_SRC)
LIB_OBJ=$(patsubst %.c,$(OBJDIR)/%.o,$(notdir $(LIB_SRC)))
MAIN_OBJ=$(OBJDIR)/main.o
OBJ=$(LIB_OBJ) $(MAIN_OBJ)
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
$(OBJDIR)/tokenize.o: lang/tokenize.c
$(OBJDIR)/vocab.o: lang/vocab.c
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
$(OBJDIR)/lm_train.o: lang/lm_train.c
$(OBJDIR)/lm_serialize.o: lang/lm_serialize.c
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

.PHONY: clean mrproper test gui run-gui all

clean:
	rm -rf $(OBJDIR)

mrproper: clean
	rm -rf $(EXEC)
	rm -rf $(TEST)
	rm -rf $(GUI)
	rm -rf *~
