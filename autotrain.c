/*
**	Opinionated Target-oriented train entry (AT1+). Teaching CLI stays ./ann.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"
#include "lm.h"
#include "lm_config.h"
#include "lm_train.h"

static void usage(const char *argv0)
{
	fprintf(stderr,
	    "Usage: %s [--out PATH] [--data PATH] [--help]\n"
	    "  AT1 skeleton: probes GPU/VRAM and prints the planned job.\n"
	    "  Full MVP (AT2–AT6) trains with BPE+stream+batch auto.\n",
	    argv0);
}

int main(int argc, char **argv)
{
	const char *out;
	const char *data;
	size_t fr;
	size_t tot;
	int i;

	out = "model_auto.bin";
	data = NULL;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
			usage(argv[0]);
			return 0;
		}
		if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
			out = argv[++i];
		else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc)
			data = argv[++i];
	}

	printf("autotrain AT1: skeleton ready (binary linked against lang/num)\n");
	printf("  out=%s\n", out);
	printf("  data=%s\n", data ? data : "(none; AT3 will fetch wiki or exit)");
	printf("  config D=%d heads=%d ff=%d ctx=%d blocks=%d\n",
	    LM_D_MODEL, LM_N_HEADS, LM_D_FF, LM_MAX_CTX, LM_N_BLOCKS);
	printf("  batch auto cap=%d default lr plan=0.001\n", LM_BATCH_AUTO_MAX);
	if (gpu_available() && gpu_mem_info(&fr, &tot) == 0)
		printf("  GPU %s  VRAM free %.0f / total %.0f MB\n",
		    gpu_backend_name(),
		    (double)fr / 1048576.0, (double)tot / 1048576.0);
	else
		printf("  GPU %s (no VRAM probe)\n", gpu_backend_name());
	printf("  next: AT2–AT6 wire packed BPE train + smoke generate\n");
	return 0;
}
