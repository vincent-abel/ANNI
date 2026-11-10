/*
**	anni-latent: 3D view of a latent space dump of anni-probe
**
**	anni-latent DIR [--compare DIR2] [--scene vocab|text] [--layer F]
**		[--checkpoint F] [--color N] [--search TEXT] [--select I]
**		[--shot FILE.ppm]
**
**	The layouts are computed at the first opening of a dump and cached in it.
**	Drag: orbit. Wheel: zoom. Right drag: look. WASD, Q, E: fly. Click: select
**	a point and draw its neighbours. Double-click: fly to it.
*/
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#define CIMGUI_USE_GLFW
#define CIMGUI_USE_OPENGL3
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include "cimgui.h"
#include "cimgui_impl.h"
#define GL_GLEXT_PROTOTYPES
#include <GL/glcorearb.h>
#include <GLFW/glfw3.h>
#include "npy.h"
#include "project.h"

#define PCA_DIMS	50
#define RADIUS		10.0f
#define VIEW_FAR	400.0f
#define UMAP_EPOCHS	300
#define UMAP_NEXT_EPOCHS	80	/* from the layout of the layer before */
#define NEIGHBOURS	12
#define PICK_PIXELS	14.0f
#define CONTEXT		14	/* tokens shown before a text point */
#define SEARCH_MAX	128
#define PLAY_SPEED	1.5f	/* layers per second */

static const char *vertex_src =
	"#version 330 core\n"
	"layout(location=0) in vec3 at;\n"
	"layout(location=1) in vec3 color;\n"
	"uniform mat4 view_proj;\n"
	"uniform float size;\n"
	"out vec4 tint;\n"
	"void main() {\n"
	"  gl_Position = view_proj * vec4(at, 1.0);\n"
	"  float fog = clamp(1.3 - gl_Position.w / 60.0, 0.1, 1.0);\n"
	"  tint = vec4(color * fog, 1.0);\n"
	"  gl_PointSize = clamp(size / gl_Position.w, 1.5, 24.0);\n"
	"}\n";
static const char *fragment_src =
	"#version 330 core\n"
	"in vec4 tint;\n"
	"out vec4 pixel;\n"
	"void main() {\n"
	"  vec2 d = gl_PointCoord * 2.0 - 1.0;\n"
	"  float glow = exp(-dot(d, d) * 3.0);\n"
	"  pixel = vec4(tint.rgb * glow * 0.35, 1.0);\n"
	"}\n";

enum { SCENE_VOCAB, SCENE_TEXT, N_SCENES };
enum { COLOR_KIND, COLOR_SOURCE, COLOR_POSITION, COLOR_LOSS, N_COLORS };

typedef struct	point {
	int	source;
	int	window;
	int	position;
	int	token;
	int	next;
	int	predicted;
	float	probability;
	float	nll;
} point;

/* One set of points, with a layout per layer and, to compare, a second one. */
typedef struct	scene {
	long	n;
	int	layers;
	float	*layout;	/* layers x n x 3 */
	float	*layout_b;	/* same, for the --compare dump, or NULL */
	float	*pos;		/* n x 3, what is drawn */
	float	*color;		/* n x 3 */
	point	*points;	/* text scene only */
} scene;

typedef struct	camera {
	float	yaw;
	float	pitch;
	float	distance;
	float	target[3];
} camera;

typedef struct	hit {
	int	index;
	float	score;
} hit;

static camera cam = {0.6f, 0.3f, 48.0f, {0.0f, 0.0f, 0.0f}};
static scene scenes[N_SCENES];
static char **vocab_text;
static long vocab_n;
static char sources[8][64];
static int n_sources;
static const char *dump_dir;
static int scene_id;
static float layer;
static float checkpoint;	/* 0 = first dump, 1 = --compare dump */
static int color_mode;
static char search[SEARCH_MAX];
static long search_hits;
static int selected = -1;
static int hovered = -1;
static hit neighbours[NEIGHBOURS];
static int n_neighbours;
static float *vectors;		/* the centred vectors the neighbours use */
static int vectors_scene = -1;
static int vectors_layer = -1;
static int fly_to = -1;
static double last_x;
static double last_y;
static double press_x;
static double press_y;

static void mat_mul4(float *out, const float *a, const float *b)
{
	int r;
	int c;
	int k;

	for (c = 0; c < 4; c++) {
		for (r = 0; r < 4; r++) {
			out[c * 4 + r] = 0.0f;
			for (k = 0; k < 4; k++)
				out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
		}
	}
}

static void camera_eye(float *eye)
{
	eye[0] = cam.target[0] + cam.distance * cosf(cam.pitch) * sinf(cam.yaw);
	eye[1] = cam.target[1] + cam.distance * sinf(cam.pitch);
	eye[2] = cam.target[2] + cam.distance * cosf(cam.pitch) * cosf(cam.yaw);
}

static void view_matrix(float *m, float aspect)
{
	float eye[3];
	float f[3];
	float s[3];
	float u[3];
	float proj[16];
	float view[16];
	float t = 1.0f / tanf(0.5f);
	float near = 0.1f;
	float len;
	int i;

	camera_eye(eye);
	for (i = 0; i < 3; i++)
		f[i] = (cam.target[i] - eye[i]) / cam.distance;
	s[0] = -f[2];
	s[1] = 0.0f;
	s[2] = f[0];
	len = sqrtf(s[0] * s[0] + s[2] * s[2]);
	s[0] /= len;
	s[2] /= len;
	u[0] = s[1] * f[2] - s[2] * f[1];
	u[1] = s[2] * f[0] - s[0] * f[2];
	u[2] = s[0] * f[1] - s[1] * f[0];
	memset(view, 0, sizeof(view));
	for (i = 0; i < 3; i++) {
		view[i * 4 + 0] = s[i];
		view[i * 4 + 1] = u[i];
		view[i * 4 + 2] = -f[i];
	}
	view[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
	view[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
	view[14] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
	view[15] = 1.0f;
	memset(proj, 0, sizeof(proj));
	proj[0] = t / aspect;
	proj[5] = t;
	proj[10] = -(VIEW_FAR + near) / (VIEW_FAR - near);
	proj[11] = -1.0f;
	proj[14] = -2.0f * VIEW_FAR * near / (VIEW_FAR - near);
	mat_mul4(m, proj, view);
}

/* Pixels of a point; 0 when it is behind the camera. */
static int to_screen(const float *m, const float *p, float w, float h, float *out)
{
	float x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
	float y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
	float q = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];

	if (q <= 0.05f)
		return 0;
	out[0] = (x / q * 0.5f + 0.5f) * w;
	out[1] = (0.5f - y / q * 0.5f) * h;
	return 1;
}

static GLuint compile(GLenum kind, const char *src)
{
	GLuint s = glCreateShader(kind);
	GLint ok;
	char log[1024];

	glShaderSource(s, 1, &src, NULL);
	glCompileShader(s);
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		glGetShaderInfoLog(s, sizeof(log), NULL, log);
		fprintf(stderr, "shader: %s\n", log);
		exit(1);
	}
	return s;
}

/* Undo the \n \t \\ of vocab.txt into something one line can show. */
static char *token_label(const char *line)
{
	char *out = malloc(strlen(line) + 1);
	char *o = out;

	for (; out && *line; line++) {
		if (*line == '\\' && line[1]) {
			line++;
			*o++ = *line == '\\' ? '\\' : ' ';
		} else {
			*o++ = *line;
		}
	}
	if (out)
		*o = '\0';
	return out;
}

static char **read_tokens(const char *path, long n)
{
	FILE *fp = fopen(path, "rb");
	char **out = calloc((size_t)n, sizeof(char *));
	char line[2048];
	long i = 0;

	if (!fp || !out)
		return NULL;
	while (i < n && fgets(line, sizeof(line), fp)) {
		line[strcspn(line, "\n")] = '\0';
		out[i++] = token_label(line);
	}
	while (i < n)
		out[i++] = strdup("");
	fclose(fp);
	return out;
}

static int token_kind(const char *t)
{
	int letters = 0;
	int digits = 0;
	int high = 0;
	int other = 0;
	const char *p;

	for (p = t; *p; p++) {
		unsigned char c = (unsigned char)*p;

		if (c >= 0x80)
			high++;
		else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
			letters++;
		else if (c >= '0' && c <= '9')
			digits++;
		else if (c != ' ')
			other++;
	}
	return high ? 3 : digits ? 1 : letters ? 0 : other ? 2 : 4;
}

static void ramp(float v, float *rgb)
{
	v = fmaxf(0.0f, fminf(1.0f, v));
	rgb[0] = 0.15f + 0.85f * v;
	rgb[1] = 0.35f + 0.45f * sinf(v * 3.14159f);
	rgb[2] = 1.0f - 0.85f * v;
}

static const char *point_text(const scene *s, long i)
{
	return vocab_text[s == &scenes[SCENE_TEXT] ? s->points[i].token : i];
}

static int matches(const char *text)
{
	const char *t = text;

	while (*t == ' ')
		t++;
	return strcasestr(t, search) != NULL;
}

static void paint(scene *s)
{
	static const float palette[5][3] = {
		{0.35f, 0.65f, 1.0f},	/* word; the first source */
		{1.0f, 0.75f, 0.3f},	/* digits */
		{1.0f, 0.4f, 0.5f},	/* punctuation, code */
		{0.5f, 1.0f, 0.65f},	/* accents, other scripts */
		{0.75f, 0.75f, 0.8f},	/* space */
	};
	long i;

	search_hits = 0;
	for (i = 0; i < s->n; i++) {
		float *rgb = s->color + i * 3;
		const point *p = s->points ? &s->points[i] : NULL;
		const char *text = point_text(s, i);

		if (p && color_mode == COLOR_SOURCE)
			memcpy(rgb, palette[p->source % 5], sizeof(palette[0]));
		else if (p && color_mode == COLOR_POSITION)
			ramp((float)p->position / 255.0f, rgb);
		else if (p && color_mode == COLOR_LOSS)
			ramp(p->nll / 10.0f, rgb);
		else
			memcpy(rgb, palette[token_kind(text)], sizeof(palette[0]));
		if (search[0]) {
			int on = matches(text);
			int c;

			search_hits += on;
			for (c = 0; c < 3; c++)
				rgb[c] *= on ? 4.0f : 0.06f;
		}
	}
}

/* Set s->pos from the layer and the checkpoint shown. */
static void place(scene *s)
{
	int low = (int)layer < s->layers - 1 ? (int)layer : s->layers - 1;
	int high = low + 1 < s->layers ? low + 1 : low;
	float t = s->layers > 1 ? layer - (float)low : 0.0f;
	const float *a = s->layout + (size_t)low * (size_t)s->n * 3;
	const float *b = s->layout + (size_t)high * (size_t)s->n * 3;
	long i;

	for (i = 0; i < s->n * 3; i++)
		s->pos[i] = a[i] + (b[i] - a[i]) * t;
	if (!s->layout_b || checkpoint <= 0.0f)
		return;
	a = s->layout_b + (size_t)low * (size_t)s->n * 3;
	b = s->layout_b + (size_t)high * (size_t)s->n * 3;
	for (i = 0; i < s->n * 3; i++)
		s->pos[i] += (a[i] + (b[i] - a[i]) * t - s->pos[i]) * checkpoint;
}

static int read_cache(const char *path, float *out, size_t count)
{
	FILE *fp = fopen(path, "rb");
	size_t got;

	if (!fp)
		return -1;
	got = fread(out, sizeof(float), count, fp);
	fclose(fp);
	return got == count ? 0 : -1;
}

static void write_cache(const char *path, const float *data, size_t count)
{
	FILE *fp = fopen(path, "wb");

	if (!fp)
		return;
	fwrite(data, sizeof(float), count, fp);
	fclose(fp);
}

/*
** Layout of the n x d rows of x into pos, x being changed. pos holds the start
** when `warm`; else the first PCA axes do.
*/
static int fit_layout(float *x, long n, int d, float *pos, int warm, int epochs)
{
	float *pca = malloc(sizeof(float) * (size_t)n * PCA_DIMS);
	int *idx = malloc(sizeof(int) * (size_t)n * KNN_K);
	float *dist = malloc(sizeof(float) * (size_t)n * KNN_K);
	long i;

	if (!pca || !idx || !dist)
		return -1;
	pca_fit(x, n, d, PCA_DIMS, pca);
	if (!warm) {
		for (i = 0; i < n; i++)
			memcpy(pos + i * 3, pca + i * PCA_DIMS, sizeof(float) * 3);
		layout_normalize(pos, n, RADIUS);
	}
	/* In the full space: 50 axes keep only a quarter of the true neighbours. */
	knn_graph(x, n, d, idx, dist);
	umap_layout(idx, dist, n, pos, epochs);
	layout_normalize(pos, n, RADIUS);
	free(pca);
	free(idx);
	free(dist);
	return 0;
}

/* The layers of a dump, each starting from the one before, or from `from`. */
static int fit_dump(scene *s, const char *dir, const char *file, float *out,
	const float *from)
{
	char path[1024];
	npy data;
	float *x;
	long d;
	int l;

	snprintf(path, sizeof(path), "%s/%s", dir, file);
	if (npy_open(&data, path) != 0)
		return -1;
	d = data.shape[data.ndim - 1];
	x = malloc(sizeof(float) * (size_t)s->n * (size_t)d);
	if (!x)
		return -1;
	for (l = 0; l < s->layers; l++) {
		float *pos = out + (size_t)l * (size_t)s->n * 3;
		int warm = from != NULL || l > 0;

		fprintf(stderr, "layout of %s, layer %d/%d\n", file, l + 1, s->layers);
		if (npy_read_f32(&data, (long)l * s->n * d, s->n * d, x) != 0)
			return -1;
		if (from)
			memcpy(pos, from + (size_t)l * (size_t)s->n * 3,
			    sizeof(float) * 3 * (size_t)s->n);
		else if (l > 0)
			memcpy(pos, pos - (size_t)s->n * 3, sizeof(float) * 3 * (size_t)s->n);
		if (fit_layout(x, s->n, (int)d, pos, warm,
		    warm ? UMAP_NEXT_EPOCHS : UMAP_EPOCHS) != 0)
			return -1;
	}
	free(x);
	npy_close(&data);
	return 0;
}

static int load_layout(scene *s, const char *dir, const char *data_file,
	const char *cache, float **out, const float *from)
{
	char path[1024];
	size_t count = (size_t)s->layers * (size_t)s->n * 3;

	*out = malloc(sizeof(float) * count);
	if (!*out)
		return -1;
	snprintf(path, sizeof(path), "%s/%s", dir, cache);
	if (read_cache(path, *out, count) == 0) {
		int l;

		for (l = 0; l < s->layers; l++)
			layout_normalize(*out + (size_t)l * s->n * 3, s->n, RADIUS);
		return 0;
	}
	if (nice(10) == -1)
		fprintf(stderr, "anni-latent: could not lower the priority\n");
	if (fit_dump(s, dir, data_file, *out, from) != 0)
		return -1;
	write_cache(path, *out, count);
	return 0;
}

static int source_id(const char *label)
{
	int i;

	for (i = 0; i < n_sources; i++) {
		if (strcmp(sources[i], label) == 0)
			return i;
	}
	if (n_sources == 8)
		return 7;
	snprintf(sources[n_sources], sizeof(sources[0]), "%s", label);
	return n_sources++;
}

static int read_points(scene *s, const char *dir)
{
	char path[1024];
	char label[64];
	FILE *fp;
	long i;

	snprintf(path, sizeof(path), "%s/points.tsv", dir);
	fp = fopen(path, "rb");
	s->points = calloc((size_t)s->n, sizeof(point));
	if (!fp || !s->points)
		return -1;
	for (i = 0; i < s->n; i++) {
		point *p = &s->points[i];

		if (fscanf(fp, "%63s %d %d %d %d %d %f %f", label, &p->window,
		    &p->position, &p->token, &p->next, &p->predicted,
		    &p->probability, &p->nll) != 8)
			return -1;
		p->source = source_id(label);
	}
	fclose(fp);
	return 0;
}

static int open_scene(int id, const char *dir, const char *compare)
{
	static const char *files[N_SCENES] = {"vocab.npy", "states.npy"};
	static const char *caches[N_SCENES] = {"layout-vocab.f32", "layout-text.f32"};
	scene *s = &scenes[id];
	char path[1024];
	npy data;

	snprintf(path, sizeof(path), "%s/%s", dir, files[id]);
	if (npy_open(&data, path) != 0)
		return -1;
	s->layers = data.ndim == 3 ? (int)data.shape[0] : 1;
	s->n = data.shape[data.ndim - 2];
	npy_close(&data);
	s->pos = malloc(sizeof(float) * 3 * (size_t)s->n);
	s->color = malloc(sizeof(float) * 3 * (size_t)s->n);
	if (!s->pos || !s->color)
		return -1;
	if (id == SCENE_TEXT && read_points(s, dir) != 0)
		return -1;
	if (load_layout(s, dir, files[id], caches[id], &s->layout, NULL) != 0)
		return -1;
	if (compare && load_layout(s, compare, files[id], caches[id], &s->layout_b,
	    s->layout) != 0)
		return -1;
	return 0;
}

/* The vectors behind the scene shown, centred, for nearest neighbours. */
static int load_vectors(const scene *s, long *d_out)
{
	static long d;
	int id = (int)(s - scenes);
	int l = (int)(layer + 0.5f);
	char path[1024];
	double *mean;
	npy data;
	long r;
	long c;

	*d_out = d;
	if (vectors && vectors_scene == id && vectors_layer == l)
		return 0;
	snprintf(path, sizeof(path), "%s/%s", dump_dir,
	    id == SCENE_TEXT ? "states.npy" : "vocab.npy");
	if (npy_open(&data, path) != 0)
		return -1;
	d = data.shape[data.ndim - 1];
	*d_out = d;
	free(vectors);
	vectors = malloc(sizeof(float) * (size_t)s->n * (size_t)d);
	mean = calloc((size_t)d, sizeof(double));
	if (!vectors || !mean
	    || npy_read_f32(&data, (id == SCENE_TEXT ? l : 0) * s->n * d, s->n * d,
	    vectors) != 0)
		return -1;
	npy_close(&data);
	for (r = 0; r < s->n; r++) {
		for (c = 0; c < d; c++)
			mean[c] += vectors[r * d + c];
	}
	#pragma omp parallel for schedule(static) private(c)
	for (r = 0; r < s->n; r++) {
		float *row = vectors + r * d;
		double norm = 1e-12;

		for (c = 0; c < d; c++) {
			row[c] -= (float)(mean[c] / (double)s->n);
			norm += (double)row[c] * row[c];
		}
		norm = 1.0 / sqrt(norm);
		for (c = 0; c < d; c++)
			row[c] *= (float)norm;
	}
	free(mean);
	vectors_scene = id;
	vectors_layer = l;
	return 0;
}

/* Nearest by cosine in the full space, not in the projection. */
static void find_neighbours(const scene *s, int index)
{
	float *score;
	long d;
	long r;
	int k;

	n_neighbours = 0;
	if (load_vectors(s, &d) != 0)
		return;
	score = malloc(sizeof(float) * (size_t)s->n);
	if (!score)
		return;
	#pragma omp parallel for schedule(static)
	for (r = 0; r < s->n; r++) {
		const float *a = vectors + (long)index * d;
		const float *b = vectors + r * d;
		float dot = 0.0f;
		long c;

		for (c = 0; c < d; c++)
			dot += a[c] * b[c];
		score[r] = dot;
	}
	score[index] = -2.0f;
	for (k = 0; k < NEIGHBOURS; k++) {
		long best = 0;

		for (r = 1; r < s->n; r++) {
			if (score[r] > score[best])
				best = r;
		}
		neighbours[k].index = (int)best;
		neighbours[k].score = score[best];
		score[best] = -2.0f;
		n_neighbours++;
	}
	free(score);
}

static void select_point(const scene *s, int index)
{
	selected = index;
	n_neighbours = 0;
	if (index >= 0)
		find_neighbours(s, index);
}

static void pick(const scene *s, const float *vp, float w, float h, float mx, float my)
{
	float best = PICK_PIXELS * PICK_PIXELS;
	long i;

	hovered = -1;
	for (i = 0; i < s->n; i++) {
		float at[2];
		float dx;
		float dy;

		if (!to_screen(vp, s->pos + i * 3, w, h, at))
			continue;
		dx = at[0] - mx;
		dy = at[1] - my;
		if (dx * dx + dy * dy < best) {
			best = dx * dx + dy * dy;
			hovered = (int)i;
		}
	}
}

static void on_scroll(GLFWwindow *w, double dx, double dy)
{
	(void)w;
	(void)dx;
	if (!igGetIO_Nil()->WantCaptureMouse)
		cam.distance *= powf(0.9f, (float)dy);
}

/* Left drag turns around the target, right drag turns the head. */
static void steer(GLFWwindow *w, float dt)
{
	float eye[3];
	float keep[3];
	double x;
	double y;
	float speed = cam.distance * dt;
	float fx;
	float fz;
	int free_keys = !igGetIO_Nil()->WantCaptureKeyboard;
	int mouse = !igGetIO_Nil()->WantCaptureMouse;
	int i;

	glfwGetCursorPos(w, &x, &y);
	camera_eye(eye);
	if (mouse && glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
		cam.yaw -= (float)(x - last_x) * 0.006f;
		cam.pitch += (float)(y - last_y) * 0.006f;
	}
	if (mouse && glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
		memcpy(keep, eye, sizeof(keep));
		cam.yaw -= (float)(x - last_x) * 0.004f;
		cam.pitch += (float)(y - last_y) * 0.004f;
		cam.pitch = fmaxf(-1.5f, fminf(1.5f, cam.pitch));
		camera_eye(eye);
		for (i = 0; i < 3; i++)
			cam.target[i] += keep[i] - eye[i];
	}
	cam.pitch = fmaxf(-1.5f, fminf(1.5f, cam.pitch));
	last_x = x;
	last_y = y;
	fx = -sinf(cam.yaw);
	fz = -cosf(cam.yaw);
	if (free_keys && glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) {
		cam.target[0] += fx * speed;
		cam.target[2] += fz * speed;
	}
	if (free_keys && glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) {
		cam.target[0] -= fx * speed;
		cam.target[2] -= fz * speed;
	}
	if (free_keys && glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) {
		cam.target[0] += fz * speed;
		cam.target[2] -= fx * speed;
	}
	if (free_keys && glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) {
		cam.target[0] -= fz * speed;
		cam.target[2] += fx * speed;
	}
	if (free_keys && glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS)
		cam.target[1] += speed;
	if (free_keys && glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS)
		cam.target[1] -= speed;
}

static void clicks(GLFWwindow *w, const scene *s)
{
	static int was_down;
	static double last_click;
	int down = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
	double now = glfwGetTime();

	if (down && !was_down)
		glfwGetCursorPos(w, &press_x, &press_y);
	if (!down && was_down && !igGetIO_Nil()->WantCaptureMouse
	    && fabs(last_x - press_x) + fabs(last_y - press_y) < 5.0) {
		select_point(s, hovered);
		if (hovered >= 0 && now - last_click < 0.35)
			fly_to = hovered;
		last_click = now;
	}
	was_down = down;
}

static void flight(const scene *s, float dt)
{
	const float *p;
	float pull = fminf(1.0f, dt * 4.0f);
	float gap = 0.0f;
	int i;

	if (fly_to < 0 || fly_to >= s->n)
		return;
	p = s->pos + (size_t)fly_to * 3;
	for (i = 0; i < 3; i++) {
		gap += fabsf(p[i] - cam.target[i]);
		cam.target[i] += (p[i] - cam.target[i]) * pull;
	}
	cam.distance += (6.0f - cam.distance) * pull;
	if (gap < 0.02f)
		fly_to = -1;
}

static void text_row(const char *name, const scene *s, int i, float score)
{
	if (score > -2.0f)
		igText("%s%s  %.3f", name, point_text(s, i), score);
	else
		igText("%s%s", name, point_text(s, i));
}

/* The text before a point of the text scene, on one line. */
static void context_of(const scene *s, int i, char *out, size_t size)
{
	int first = i - s->points[i].position;
	int from = i - CONTEXT > first ? i - CONTEXT : first;
	size_t used = 0;
	int k;

	out[0] = '\0';
	for (k = from; k <= i && used + 1 < size; k++)
		used += (size_t)snprintf(out + used, size - used, "%s%s",
		    k == i ? " >>" : "", vocab_text[s->points[k].token]);
}

static void point_details(const scene *s, int i)
{
	char context[1024];
	const point *p;

	if (!s->points) {
		igText("token %d: \"%s\"", i, vocab_text[i]);
		return;
	}
	p = &s->points[i];
	context_of(s, i, context, sizeof(context));
	igPushTextWrapPos(420.0f);
	igTextUnformatted(context, NULL);
	igPopTextWrapPos();
	igText("%s, window %d, position %d", sources[p->source], p->window, p->position);
	igText("next \"%s\", NLL %.2f", vocab_text[p->next], p->nll);
	igText("predicted \"%s\", %.0f%%", vocab_text[p->predicted], p->probability * 100.0f);
}

static void overlay(const scene *s, const float *vp, float w, float h)
{
	ImDrawList *dl = igGetForegroundDrawList_Nil();
	ImVec2_c from;
	ImVec2_c to;
	float at[2];
	int k;

	if (selected < 0 || !to_screen(vp, s->pos + (size_t)selected * 3, w, h, at))
		return;
	from.x = at[0];
	from.y = at[1];
	for (k = 0; k < n_neighbours; k++) {
		const char *t = point_text(s, neighbours[k].index);

		if (!to_screen(vp, s->pos + (size_t)neighbours[k].index * 3, w, h, at))
			continue;
		to.x = at[0];
		to.y = at[1];
		ImDrawList_AddLine(dl, from, to, 0x60ffffffu, 1.0f, 0);
		ImDrawList_AddText_Vec2(dl, to, 0xc0ffffffu, t, NULL);
	}
	ImDrawList_AddCircle(dl, from, 7.0f, 0xff40e0ffu, 0, 2.0f, 0);
	ImDrawList_AddText_Vec2(dl, from, 0xff40e0ffu, point_text(s, selected), NULL);
}

static void panel(scene *s, int *playing, float dt)
{
	static const char *scene_names[N_SCENES] = {"Vocabulary", "Text through the layers"};
	static const char *color_names[N_COLORS] = {"Token kind", "Source", "Position", "Loss"};
	ImVec2_c size = {0.0f, 0.0f};
	ImVec2_c pos = {14.0f, 14.0f};
	ImVec2_c pivot = {0.0f, 0.0f};
	int redraw = 0;
	int id = scene_id;
	int k;

	igSetNextWindowPos(pos, ImGuiCond_FirstUseEver, pivot);
	igSetNextWindowSize((ImVec2_c){310.0f, 0.0f}, ImGuiCond_FirstUseEver);
	igBegin("Latent space", NULL, 0);
	if (scenes[SCENE_TEXT].n) {
		if (igCombo_Str_arr("Scene", &id, scene_names, N_SCENES, -1) && id != scene_id) {
			scene_id = id;
			s = &scenes[id];
			selected = hovered = fly_to = -1;
			n_neighbours = 0;
			layer = 0.0f;
			redraw = 1;
		}
	}
	igText("%ld points, %d layer%s", s->n, s->layers, s->layers > 1 ? "s" : "");
	if (s->layers > 1) {
		if (igSliderFloat("Layer", &layer, 0.0f, (float)(s->layers - 1), "%.1f", 0))
			*playing = 0;
		if (igButton(*playing ? "Pause" : "Play", size))
			*playing = !*playing;
		if (*playing) {
			layer += dt * PLAY_SPEED;
			if (layer >= (float)(s->layers - 1))
				layer = 0.0f;
		}
	}
	if (s->layout_b)
		igSliderFloat("Checkpoint", &checkpoint, 0.0f, 1.0f, "%.2f", 0);
	if (s->points && igCombo_Str_arr("Colour", &color_mode, color_names, N_COLORS, -1))
		redraw = 1;
	if (igInputText("Search", search, sizeof(search), 0, NULL, NULL))
		redraw = 1;
	if (search[0])
		igText("%ld matches", search_hits);
	if (redraw) {
		paint(s);
		select_point(s, -1);
	}
	if (selected >= 0) {
		igSeparator();
		point_details(s, selected);
		igText("Nearest in the full space:");
		for (k = 0; k < n_neighbours; k++)
			text_row("  ", s, neighbours[k].index, neighbours[k].score);
	}
	igSeparator();
	igText("%.0f fps", igGetIO_Nil()->Framerate);
	igEnd();
}

static void save_shot(const char *path, int w, int h)
{
	unsigned char *px = malloc((size_t)w * (size_t)h * 3);
	FILE *fp = fopen(path, "wb");
	int y;

	if (!px || !fp)
		return;
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px);
	fprintf(fp, "P6\n%d %d\n255\n", w, h);
	for (y = h - 1; y >= 0; y--)
		fwrite(px + (size_t)y * (size_t)w * 3, 3, (size_t)w, fp);
	fclose(fp);
	free(px);
}

static const char *option(int argc, char **argv, const char *name)
{
	int i;

	for (i = 2; i + 1 < argc; i++) {
		if (strcmp(argv[i], name) == 0)
			return argv[i + 1];
	}
	return NULL;
}

int main(int argc, char **argv)
{
	const char *shot = option(argc, argv, "--shot");
	const char *compare = option(argc, argv, "--compare");
	const char *arg;
	char path[1024];
	GLFWwindow *win;
	GLuint vao;
	GLuint vbo[2];
	GLuint prog;
	GLint u_view;
	GLint u_size;
	double before;
	int shown = -1;
	int playing = 0;
	int frames = 0;
	npy table;

	if (argc < 2) {
		fprintf(stderr, "usage: anni-latent DIR [--compare DIR2] [--scene vocab|text] "
		    "[--layer F] [--color N] [--search TEXT] [--select I] [--shot FILE.ppm]\n");
		return 2;
	}
	dump_dir = argv[1];
	snprintf(path, sizeof(path), "%s/vocab.npy", dump_dir);
	if (npy_open(&table, path) != 0) {
		fprintf(stderr, "anni-latent: no dump in %s\n", dump_dir);
		return 1;
	}
	vocab_n = table.shape[0];
	npy_close(&table);
	snprintf(path, sizeof(path), "%s/vocab.txt", dump_dir);
	vocab_text = read_tokens(path, vocab_n);
	if (!vocab_text || open_scene(SCENE_VOCAB, dump_dir, compare) != 0) {
		fprintf(stderr, "anni-latent: cannot read %s\n", dump_dir);
		return 1;
	}
	snprintf(path, sizeof(path), "%s/states.npy", dump_dir);
	if (access(path, R_OK) == 0 && open_scene(SCENE_TEXT, dump_dir, compare) != 0) {
		fprintf(stderr, "anni-latent: cannot read the text dump of %s\n", dump_dir);
		return 1;
	}
	arg = option(argc, argv, "--scene");
	if (arg && strcmp(arg, "text") == 0 && scenes[SCENE_TEXT].n)
		scene_id = SCENE_TEXT;
	arg = option(argc, argv, "--layer");
	if (arg)
		layer = (float)atof(arg);
	arg = option(argc, argv, "--checkpoint");
	if (arg)
		checkpoint = (float)atof(arg);
	arg = option(argc, argv, "--color");
	if (arg)
		color_mode = atoi(arg);
	arg = option(argc, argv, "--search");
	if (arg)
		snprintf(search, sizeof(search), "%s", arg);
	if (!glfwInit())
		return 1;
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	win = glfwCreateWindow(1400, 900, "anni-latent", NULL, NULL);
	if (!win) {
		fprintf(stderr, "anni-latent: no OpenGL 3.3 window\n");
		return 1;
	}
	glfwMakeContextCurrent(win);
	glfwSwapInterval(1);
	igCreateContext(NULL);
	igGetIO_Nil()->IniFilename = NULL;
	ImGui_ImplGlfw_InitForOpenGL(win, true);
	ImGui_ImplOpenGL3_Init("#version 330 core");
	glfwSetScrollCallback(win, on_scroll);

	prog = glCreateProgram();
	glAttachShader(prog, compile(GL_VERTEX_SHADER, vertex_src));
	glAttachShader(prog, compile(GL_FRAGMENT_SHADER, fragment_src));
	glLinkProgram(prog);
	u_view = glGetUniformLocation(prog, "view_proj");
	u_size = glGetUniformLocation(prog, "size");
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(2, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(1);

	before = glfwGetTime();
	while (!glfwWindowShouldClose(win)) {
		scene *s = &scenes[scene_id];
		double now = glfwGetTime();
		float dt = (float)(now - before);
		ImGuiIO *io;
		float vp[16];
		int w;
		int h;

		before = now;
		glfwPollEvents();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		igNewFrame();
		io = igGetIO_Nil();
		if (shown != scene_id) {
			paint(s);
			shown = scene_id;
			arg = option(argc, argv, "--select");
			if (arg && frames == 0) {
				place(s);
				select_point(s, atoi(arg));
			}
		}
		panel(s, &playing, dt);
		s = &scenes[scene_id];
		place(s);
		steer(win, dt);
		flight(s, dt);
		glfwGetFramebufferSize(win, &w, &h);
		view_matrix(vp, (float)w / (float)(h > 0 ? h : 1));
		if (!io->WantCaptureMouse)
			pick(s, vp, (float)w, (float)h, (float)last_x, (float)last_y);
		else
			hovered = -1;
		clicks(win, s);
		overlay(s, vp, (float)w, (float)h);
		if (hovered >= 0) {
			igBeginTooltip();
			point_details(s, hovered);
			igEndTooltip();
		}
		igRender();
		glViewport(0, 0, w, h);
		glClearColor(0.01f, 0.012f, 0.03f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glUseProgram(prog);
		glUniformMatrix4fv(u_view, 1, GL_FALSE, vp);
		glUniform1f(u_size, 120.0f);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, s->n * 12, s->pos, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, s->n * 12, s->color, GL_DYNAMIC_DRAW);
		glBindVertexArray(vao);
		glDrawArrays(GL_POINTS, 0, (GLsizei)s->n);
		glDisable(GL_BLEND);
		ImGui_ImplOpenGL3_RenderDrawData(igGetDrawData());
		if (shot && frames >= 4) {
			save_shot(shot, w, h);
			break;
		}
		frames++;
		glfwSwapBuffers(win);
	}
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	igDestroyContext(NULL);
	glfwTerminate();
	return 0;
}
