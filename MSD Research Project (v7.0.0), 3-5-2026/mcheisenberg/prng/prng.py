class SplitMix64:
    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
        return (z ^ (z >> 31)) & ((1 << 64) - 1)

if __name__ == "__main__":
    rng = SplitMix64(0)
    for i in range(10):
        x = rng.next()
        print(f"[{i}.] {hex(x)}")
