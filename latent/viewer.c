/*
**	anni-latent: 3D view of a latent space dump of anni-probe
**
**	anni-latent DIR [--shot FILE.ppm]
*/
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#define CIMGUI_USE_GLFW
#define CIMGUI_USE_OPENGL3
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

typedef struct	scene {
	long	n;
	float	*pos;		/* n x 3 */
	float	*color;		/* n x 3 */
	char	**text;
} scene;

typedef struct	camera {
	float	yaw;
	float	pitch;
	float	distance;
	float	target[3];
} camera;

static camera cam = {0.6f, 0.3f, 32.0f, {0.0f, 0.0f, 0.0f}};
static double last_x;
static double last_y;

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

	eye[0] = cam.target[0] + cam.distance * cosf(cam.pitch) * sinf(cam.yaw);
	eye[1] = cam.target[1] + cam.distance * sinf(cam.pitch);
	eye[2] = cam.target[2] + cam.distance * cosf(cam.pitch) * cosf(cam.yaw);
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

static char **read_lines(const char *path, long n)
{
	FILE *fp = fopen(path, "rb");
	char **out = calloc((size_t)n, sizeof(char *));
	char line[2048];
	long i = 0;

	if (!fp || !out)
		return out;
	while (i < n && fgets(line, sizeof(line), fp)) {
		line[strcspn(line, "\n")] = '\0';
		out[i++] = strdup(line);
	}
	fclose(fp);
	return out;
}

/* Colour of a token by what it is made of. */
static void token_color(const char *t, float *rgb)
{
	static const float kinds[5][3] = {
		{0.35f, 0.65f, 1.0f},	/* word */
		{1.0f, 0.75f, 0.3f},	/* digits */
		{1.0f, 0.4f, 0.5f},	/* punctuation, code */
		{0.5f, 1.0f, 0.65f},	/* accents, other scripts */
		{0.75f, 0.75f, 0.8f},	/* space */
	};
	int letters = 0;
	int digits = 0;
	int high = 0;
	int other = 0;
	const char *p;
	int kind;

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
	kind = high ? 3 : digits ? 1 : letters ? 0 : other ? 2 : 4;
	memcpy(rgb, kinds[kind], sizeof(kinds[0]));
}

/* Quick layout: the three main axes of the table, cached in DIR. */
static int vocab_scene(scene *s, const char *dir)
{
	char path[1024];
	npy table;
	float *x;
	float *pca;
	long i;
	FILE *fp;

	snprintf(path, sizeof(path), "%s/vocab.npy", dir);
	if (npy_open(&table, path) != 0)
		return -1;
	s->n = table.shape[0];
	s->pos = malloc(sizeof(float) * 3 * (size_t)s->n);
	s->color = malloc(sizeof(float) * 3 * (size_t)s->n);
	snprintf(path, sizeof(path), "%s/vocab.txt", dir);
	s->text = read_lines(path, s->n);
	snprintf(path, sizeof(path), "%s/layout-vocab-pca.f32", dir);
	fp = fopen(path, "rb");
	if (fp) {
		size_t got = fread(s->pos, sizeof(float) * 3, (size_t)s->n, fp);

		fclose(fp);
		if (got != (size_t)s->n)
			return -1;
	} else {
		int d = (int)table.shape[1];

		x = malloc(sizeof(float) * (size_t)s->n * (size_t)d);
		pca = malloc(sizeof(float) * (size_t)s->n * PCA_DIMS);
		if (!x || !pca || npy_read_f32(&table, 0, s->n * d, x) != 0)
			return -1;
		fprintf(stderr, "PCA of %ld tokens\n", s->n);
		pca_fit(x, s->n, d, PCA_DIMS, pca);
		for (i = 0; i < s->n; i++)
			memcpy(s->pos + i * 3, pca + i * PCA_DIMS, sizeof(float) * 3);
		layout_normalize(s->pos, s->n, RADIUS);
		free(x);
		free(pca);
		fp = fopen(path, "wb");
		if (fp) {
			fwrite(s->pos, sizeof(float) * 3, (size_t)s->n, fp);
			fclose(fp);
		}
	}
	npy_close(&table);
	for (i = 0; i < s->n; i++)
		token_color(s->text[i] ? s->text[i] : "", s->color + i * 3);
	return 0;
}

static void on_scroll(GLFWwindow *w, double dx, double dy)
{
	(void)w;
	(void)dx;
	if (!igGetIO_Nil()->WantCaptureMouse)
		cam.distance *= powf(0.9f, (float)dy);
}

static void orbit(GLFWwindow *w)
{
	double x;
	double y;

	glfwGetCursorPos(w, &x, &y);
	if (!igGetIO_Nil()->WantCaptureMouse
	    && glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
		cam.yaw -= (float)(x - last_x) * 0.006f;
		cam.pitch += (float)(y - last_y) * 0.006f;
		cam.pitch = fmaxf(-1.5f, fminf(1.5f, cam.pitch));
	}
	last_x = x;
	last_y = y;
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

int main(int argc, char **argv)
{
	const char *shot = NULL;
	GLFWwindow *win;
	scene s;
	GLuint vao;
	GLuint vbo[2];
	GLuint prog;
	GLint u_view;
	GLint u_size;
	int frames = 0;
	int i;

	if (argc < 2) {
		fprintf(stderr, "usage: anni-latent DIR [--shot FILE.ppm]\n");
		return 2;
	}
	for (i = 2; i + 1 < argc; i++) {
		if (strcmp(argv[i], "--shot") == 0)
			shot = argv[i + 1];
	}
	memset(&s, 0, sizeof(s));
	if (vocab_scene(&s, argv[1]) != 0) {
		fprintf(stderr, "anni-latent: cannot read %s\n", argv[1]);
		return 1;
	}
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
	glBufferData(GL_ARRAY_BUFFER, s.n * 12, s.pos, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
	glBufferData(GL_ARRAY_BUFFER, s.n * 12, s.color, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(1);

	while (!glfwWindowShouldClose(win)) {
		float vp[16];
		int w;
		int h;

		glfwPollEvents();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		igNewFrame();
		orbit(win);
		igBegin("Latent space", NULL, 0);
		igText("%ld tokens, quick PCA layout", s.n);
		igText("%.0f fps", igGetIO_Nil()->Framerate);
		igEnd();
		igRender();
		glfwGetFramebufferSize(win, &w, &h);
		glViewport(0, 0, w, h);
		glClearColor(0.01f, 0.012f, 0.03f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glUseProgram(prog);
		view_matrix(vp, (float)w / (float)(h > 0 ? h : 1));
		glUniformMatrix4fv(u_view, 1, GL_FALSE, vp);
		glUniform1f(u_size, 120.0f);
		glBindVertexArray(vao);
		glDrawArrays(GL_POINTS, 0, (GLsizei)s.n);
		glDisable(GL_BLEND);
		ImGui_ImplOpenGL3_RenderDrawData(igGetDrawData());
		if (shot && ++frames == 4) {
			save_shot(shot, w, h);
			break;
		}
		glfwSwapBuffers(win);
	}
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	igDestroyContext(NULL);
	glfwTerminate();
	return 0;
}
