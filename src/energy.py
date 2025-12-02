def energy(self, config):
    """Compute number of attacking queen pairs."""
    E = 0
    Q = len(config)
    for i in range(Q):
        for j in range(i+1, Q):
            if self.queens_attack(config[i], config[j]):
                E += 1
    return E