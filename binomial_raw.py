from enum import Enum
import math
import matplotlib.pyplot as plt

class PayoffType(Enum):
    Call = 0
    Put = 1

class EuropeanOption():
    def __init__(self, expiry, strike, payoffType):
        self.expiry = expiry
        self.strike = strike
        self.payoffType = payoffType
    def payoff(self, S):
        if self.payoffType == PayoffType.Call:
            return max(S - self.strike, 0)
        elif self.payoffType == PayoffType.Put:
            return max(self.strike - S, 0)
        else:
            raise Exception("payoffType not supported: ", self.payoffType)
    def valueAtNode(self, t, S, continuation):
        return continuation

class AmericanOption():
    def __init__(self, expiry, strike, payoffType):
        self.expiry = expiry
        self.strike = strike
        self.payoffType = payoffType
    def payoff(self, S):
        if self.payoffType == PayoffType.Call:
            return max(S - self.strike, 0)
        elif self.payoffType == PayoffType.Put:
            return max(self.strike - S, 0)
        else:
            raise Exception("payoffType not supported: ", self.payoffType)
    def valueAtNode(self, t, S, continuation):
        return max(self.payoff(S), continuation)

# Black-Scholes analytic pricer
def cnorm(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bsPrice(S, r, vol, T, strike, payoffType):
    fwd = S * math.exp(r * T)
    stdev = vol * math.sqrt(T)
    d1 = math.log(fwd / strike) / stdev + stdev / 2
    d2 = d1 - stdev
    if payoffType == PayoffType.Call:
        return math.exp(-r * T) * (fwd * cnorm(d1) - cnorm(d2) * strike)
    elif payoffType == PayoffType.Put:
        return math.exp(-r * T) * (strike * cnorm(-d2) - cnorm(-d1) * fwd)
    else:
        raise Exception("not supported payoff type", payoffType)

def crrBinomial(S, r, vol, trade, n):
    t = trade.expiry / n
    b = math.exp(vol * vol * t+r*t) + math.exp(-r * t)
    u = (b + math.sqrt(b*b - 4)) / 2
    p = (math.exp(r * t) - (1/u)) / (u - 1/u)
    # d = 1 / u
    # set up the last time slice, there are n+1 nodes at the last time slice
    vs = [trade.payoff( S * u**(n-i-i)) for i in range(n+1)]
    # iterate backward
    for i in range(n-1, -1, -1):
        # calculate the value of each node at time slide i, there are i nodes
        for j in range(i+1):
            vs[j] = math.exp(-r * t) * (vs[j] * p + vs[j+1] * (1-p))
    return vs[0]

def crrBinomialAmer(S, r, vol, trade, n):
    t = trade.expiry / n
    b = math.exp(vol * vol * t+r*t) + math.exp(-r * t)
    u = (b + math.sqrt(b*b - 4)) / 2
    p = (math.exp(r * t) - (1/u)) / (u - 1/u)
    # d = 1 / u
    # set up the last time slice, there are n+1 nodes at the last time slice
    vs = [trade.payoff( S * u**(n-i-i)) for i in range(n+1)]
    # iterate backward
    for i in range(n-1, -1, -1):
        # calculate the value of each node at time slide i, there are i nodes
        for j in range(i+1):
            vs[j] = max(math.exp(-r * t) * (vs[j] * p + vs[j+1] * (1-p)), trade.payoff(S * u**(i-j-j)))
    return vs[0]

def crrBinomialG(S, r, vol, trade, n):
    t = trade.expiry / n
    b = math.exp(vol * vol * t+r*t) + math.exp(-r * t)
    u = (b + math.sqrt(b*b - 4)) / 2
    p = (math.exp(r * t) - (1/u)) / (u - 1/u)
    # d = 1 / u
    # set up the last time slice, there are n+1 nodes at the last time slice
    vs = [trade.payoff( S * u**(n-i-i)) for i in range(n+1)]
    # iterate backward
    for i in range(n-1, -1, -1):
        # calculate the value of each node at time slide i, there are i nodes
        for j in range(i+1):
            nodeS = S * u**(i-j-j)
            continuation = math.exp(-r * t) * (vs[j] * p + vs[j+1] * (1-p))
            vs[j] = trade.valueAtNode(t*i, nodeS, continuation)
    return vs[0]

def crrCalib(r, vol, t):
    b = math.exp(vol * vol * t + r * t) + math.exp(-r * t)
    u = (b + math.sqrt(b * b - 4)) / 2
    p = (math.exp(r * t) - (1 / u)) / (u - 1 / u)
    return (u, 1/u, p)

def jrrnCalib(r, vol, t):
    u = math.exp((r - vol * vol / 2) * t + vol * math.sqrt(t))
    d = math.exp((r - vol * vol / 2) * t - vol * math.sqrt(t))
    p = (math.exp(r * t) - d) / (u - d)
    return (u, d, p)

def jreqCalib(r, vol, t):
    u = math.exp((r - vol * vol / 2) * t + vol * math.sqrt(t))
    d = math.exp((r - vol * vol / 2) * t - vol * math.sqrt(t))
    return (u, d, 1/2)

def tianCalib(r, vol, t):
    v = math.exp(vol * vol * t)
    u = 0.5 * math.exp(r * t) * v * (v + 1 + math.sqrt(v*v + 2*v - 3))
    d = 0.5 * math.exp(r * t) * v * (v + 1 - math.sqrt(v*v + 2*v - 3))
    p = (math.exp(r * t) - d) / (u - d)
    return (u, d, p)

# assignment1 - one step analytic, american option, gamma
# assignment1 - compare greeks stability between different bionmial models (maybe trinomial)

def binomialPricer(S, r, vol, trade, n, calib):
    t = trade.expiry / n
    (u, d, p) = calib(r, vol, t)
    # set up the last time slice, there are n+1 nodes at the last time slice
    vs = [trade.payoff(S * u ** (n - i) * d ** i) for i in range(n + 1)]
    # iterate backward
    for i in range(n - 1, -1, -1):
        # calculate the value of each node at time slide i, there are i nodes
        for j in range(i + 1):
            nodeS = S * u ** (i - j) * d ** j
            continuation = math.exp(-r * t) * (vs[j] * p + vs[j + 1] * (1 - p))
            vs[j] = trade.valueAtNode(t * i, nodeS, continuation)
    return vs[0]

def distanceKandNode(k, S, r, vol, T, n):
    t = T / n
    b = math.exp(vol * vol * t + r * t) + math.exp(-r * t)
    u = (b + math.sqrt(b * b - 4)) / 2
    nodes = [S * u**(n-i-i) for i in range(n+1)]
    dis = map(lambda s: abs(s - k), nodes)
    return min(dis)

# assignment1 - one step analytic, american option, gamma
# assignment1 - fixe point convertor
# assignment1 - compare greeks stability between different bionmial models (maybe trinomial)


if __name__ == "__main__":
    opt = EuropeanOption(1, 105, PayoffType.Call)
    S, r, vol = 100, 0.01, 0.2
    bsprc = bsPrice(S, r, vol, opt.expiry, opt.strike, opt.payoffType)
    print("bsPrice = \t ", bsprc)
    n = 300
    crrErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, crrCalib) - bsprc)) for i in range(1, n)]
    jrrnErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, jrrnCalib) - bsprc)) for i in range(1, n)]
    jreqErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, jreqCalib) - bsprc)) for i in range(1, n)]
    tianErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, tianCalib) - bsprc)) for i in range(1, n)]
    plt.plot(range(1, n), crrErrs, label = "crr")
    plt.plot(range(1, n), jrrnErrs, label = "jrrn")
    plt.plot(range(1, n), jreqErrs, label = "jreq")
    plt.plot(range(1, n), tianErrs, label="tian"), plt.legend()
    plt.savefig('../figs/triError.eps', format='eps')

    # errs = [math.log(abs(crrBinomial(S, r, vol, opt, i) - bsprc)) for i in range(1, n)]
    # dis = [distanceKandNode(opt.strike, S, r, vol, opt.expiry, i) for i in range(1, n)]
    # plt.plot(range(1, n), errs, 'r')
    # plt.plot(range(1, n), dis, 'g')
    # plt.show()
    #
    # pass
    #
    # euroPrc = []
    # amerPrc = []
    # ks = range(50, 150)
    # for k in ks:
    #     opt = EuropeanOption(1, float(k), PayoffType.Call)
    #     euroPrc.append(crrBinomial(S, r, vol, opt, 300))
    #     amerPrc.append(crrBinomialAmer(S, r, vol, opt, 300))
    #
    # plt.plot(ks, euroPrc, 'r')
    # plt.plot(ks, amerPrc, 'g')
    # plt.show()
    #
    # ks = range(50, 150)
    # euroPrc = []
    # amerPrc = []
    # for k in ks:
    #     euroOpt = EuropeanOption(1, float(k), PayoffType.Put)
    #     amerOpt = AmericanOption(1, float(k), PayoffType.Put)
    #     euroPrc.append(crrBinomialG(S, r, vol, euroOpt, 300))
    #     amerPrc.append(crrBinomialG(S, r, vol, amerOpt, 300))
    #
    # plt.plot(ks, euroPrc, 'r')
    # plt.plot(ks, amerPrc, 'g')
    # plt.show()
    #
    # print("done")
