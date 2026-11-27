/*
 * membw.c: attained memory bandwidth of one device, multi-threaded.
 *
 * During decoding, every token reads each weight once, so the read test is the
 * number that bounds time per token. The copy test is reported for comparison
 * with STREAM-style figures.
 *
 * Build and run (macOS, Linux, or Android in Termux after `pkg install clang`):
 *     cc -O3 -pthread -o membw membw.c && ./membw
 * Options:
 *     -m MiB      buffer size per array (default 1024; keep it well above the caches)
 *     -t threads  worker threads (default: online CPUs; on phones also try the
 *                 number of big cores, since little cores can drag the total down)
 *     -r reps     repetitions (default 15). Best = peak; last = after any thermal
 *                 throttling has kicked in.
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    const uint64_t *src;
    uint64_t *dst;
    size_t n;
    int copy;
    uint64_t sum;
} job_t;

static void *work(void *p) {
    job_t *j = (job_t *)p;
    if (j->copy) {
        memcpy(j->dst, j->src, j->n * sizeof(uint64_t));
    } else {
        uint64_t s = 0;
        for (size_t i = 0; i < j->n; i++) s += j->src[i];
        j->sum = s;
    }
    return NULL;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* Run one pass over the buffer with `nt` threads; return seconds and add to *chk. */
static double pass(const uint64_t *src, uint64_t *dst, size_t n, int nt, int copy,
                   uint64_t *chk) {
    pthread_t th[256];
    job_t jobs[256];
    size_t per = n / (size_t)nt;
    double t0 = now();
    for (int i = 0; i < nt; i++) {
        size_t off = per * (size_t)i;
        jobs[i].src = src + off;
        jobs[i].dst = dst ? dst + off : NULL;
        jobs[i].n = (i == nt - 1) ? n - off : per;
        jobs[i].copy = copy;
        jobs[i].sum = 0;
        pthread_create(&th[i], NULL, work, &jobs[i]);
    }
    for (int i = 0; i < nt; i++) {
        pthread_join(th[i], NULL);
        *chk += jobs[i].sum;
    }
    return now() - t0;
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    size_t mib = 1024;
    int nt = (int)sysconf(_SC_NPROCESSORS_ONLN);
    int reps = 15;
    int c;
    while ((c = getopt(argc, argv, "m:t:r:")) != -1) {
        if (c == 'm') mib = (size_t)atol(optarg);
        else if (c == 't') nt = atoi(optarg);
        else if (c == 'r') reps = atoi(optarg);
        else {
            fprintf(stderr, "usage: %s [-m MiB] [-t threads] [-r reps]\n", argv[0]);
            return 1;
        }
    }
    if (nt < 1) nt = 1;
    if (nt > 256) nt = 256;
    if (reps < 1) reps = 1;
    if (reps > 1000) reps = 1000;

    size_t n = mib * 1024 * 1024 / sizeof(uint64_t);
    uint64_t *a = malloc(n * sizeof(uint64_t));
    uint64_t *b = malloc(n * sizeof(uint64_t));
    if (!a || !b) {
        fprintf(stderr, "could not allocate 2 x %zu MiB; lower -m\n", mib);
        return 1;
    }
    for (size_t i = 0; i < n; i++) a[i] = i; /* fault every page in */
    memset(b, 0, n * sizeof(uint64_t));

    uint64_t chk = 0;
    double bytes = (double)n * sizeof(uint64_t);
    double rd[1000], cp[1000];
    pass(a, NULL, n, nt, 0, &chk); /* warm-up */
    for (int r = 0; r < reps; r++) {
        rd[r] = bytes / pass(a, NULL, n, nt, 0, &chk) / 1e9;
        cp[r] = 2.0 * bytes / pass(a, b, n, nt, 1, &chk) / 1e9; /* read + write */
    }
    double rd_last = rd[reps - 1], cp_last = cp[reps - 1];
    qsort(rd, (size_t)reps, sizeof(double), cmp);
    qsort(cp, (size_t)reps, sizeof(double), cmp);

    printf("buffer %zu MiB, %d threads, %d reps\n", mib, nt, reps);
    printf("read : best %6.1f  median %6.1f  last %6.1f GB/s\n", rd[reps - 1],
           rd[reps / 2], rd_last);
    printf("copy : best %6.1f  median %6.1f  last %6.1f GB/s\n", cp[reps - 1],
           cp[reps / 2], cp_last);
    printf("{\"read_gbps\": %.1f, \"copy_gbps\": %.1f, \"threads\": %d}\n", rd[reps / 2],
           cp[reps / 2], nt);
    fprintf(stderr, "(checksum %llu)\n", (unsigned long long)chk);
    free(a);
    free(b);
    return 0;
}
