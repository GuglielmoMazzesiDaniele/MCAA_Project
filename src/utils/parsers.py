
""" Parsers used by the BOConfig to make the optimization modular"""

def exponential_parser(v):
    """
    Parse optimization vector for ExponentialScheduler.
    v[0]: beta (start_beta)
    v[1]: end_beta
    Returns: beta, start_beta, end_beta (max_iters is set separately)
    """
    beta = float(v[0].item())
    end_beta = float(v[1].item())
    return beta, beta, end_beta  # start_beta = beta


def exponential_alpha_parser(v):
    """
    Parse optimization vector for ExponentialScheduler with alpha tuning.
    v[0]: beta (start_beta)
    v[1]: alpha
    Returns: beta, start_beta, alpha
    """
    beta = float(v[0].item())
    alpha = v[1]
    return beta, beta, alpha  # start_beta = beta


def logistic_parser(v):
    """
    Parse optimization vector for LogisticScheduler.
    v[0]: beta_max
    v[1]: k (steepness parameter)
    Returns: beta, beta_max, k
    """
    beta = float(v[0].item())
    beta_max = float(v[0].item())
    k = float(v[1].item())
    return beta, beta_max, k


def constant_parser(v):
    """
    Parse optimization vector for ConstantScheduler.
    v[0]: beta
    Returns: beta, beta
    """
    beta = float(v[0].item())
    return beta, beta


def geometric_parser(v):
    """
    Parse optimization vector for GeometricScheduler.
    v[0]: beta (initial)
    v[1]: alpha (multiplicative factor)
    Returns: beta, alpha
    """
    beta = float(v[0].item())
    alpha = float(v[1].item())
    return beta, alpha


def linear_parser(v):
    """
    Parse optimization vector for LinearScheduler.
    v[0]: beta (initial)
    v[1]: a (multiplicative factor)
    v[2]: b (additive factor)
    Returns: beta, a, b
    """
    beta = float(v[0].item())
    a = float(v[1].item())
    b = float(v[2].item())
    return beta, a, b


def log_parser(v):
    """
    Parse optimization vector for LogScheduler.
    v[0]: beta (start_beta)
    v[1]: end_beta
    Returns: beta, start_beta, end_beta
    """
    beta = float(v[0].item())
    end_beta = float(v[1].item())
    return beta, beta, end_beta


def power_parser(v):
    """
    Parse optimization vector for PowerScheduler.
    v[0]: beta (initial)
    v[1]: beta_max
    v[2]: p (power exponent)
    Returns: beta, beta_max, p
    """
    beta = float(v[0].item())
    beta_max = float(v[1].item())
    p = float(v[2].item())
    return beta, beta_max, p


def step_parser(v):
    """
    Parse optimization vector for StepScheduler.
    v[0]: beta (initial)
    v[1]: gamma (multiplicative factor)
    v[2]: step_size (iterations between increases)
    Returns: beta, gamma, step_size
    """
    beta = float(v[0].item())
    gamma = float(v[1].item())
    step_size = int(v[2].item())
    return beta, gamma, step_size


def adaptive_step_parser(v):
    """
    Parse optimization vector for AdaptiveStepScheduler.
    v[0]: start_beta
    v[1]: end_beta
    v[2]: num_steps
    Returns: beta, start_beta, end_beta, num_steps
    """
    start_beta = float(v[0].item())
    end_beta = float(v[1].item())
    num_steps = int(v[2].item())
    return start_beta, start_beta, end_beta, num_steps