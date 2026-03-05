#include <stdint.h>
#include <stdio.h>

static uint64_t s[4];

static uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// splitmix64 for seeding
static uint64_t splitmix64_next(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// xoshiro256** “ss”
static uint64_t xoshiro256ss_ref(void) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;

    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 45);

    return result;
}

int main(void) {
    uint64_t seed = 0;
    // seed state using splitmix64(seed)
    for (int i = 0; i < 4; i++)
        s[i] = splitmix64_next(&seed);

    printf("Initial state: {%llu, %llu, %llu, %llu}\n",
           (unsigned long long)s[0],
           (unsigned long long)s[1],
           (unsigned long long)s[2],
           (unsigned long long)s[3]);

    for (int i = 0; i < 100; i++) {
        uint64_t x = xoshiro256ss_ref();
        printf("[%2d.] 0x %016llX == %llu\n", i, x, x);
    }
}
