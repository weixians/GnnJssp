class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]

    def append(self, adj, fea, candidate, mask, action, reward, done, logprob):
        self.adj_mb.append(adj)
        self.fea_mb.append(fea)
        self.candidate_mb.append(candidate)
        self.mask_mb.append(mask)
        self.a_mb.append(action)
        self.r_mb.append(reward)
        self.done_mb.append(done)
        self.logprobs.append(logprob)
