def exponential_parser(v):
    """
    Parse optimization vector for ExponentialScheduler.
    v[0]: beta (start_beta)
    v[1]: end_beta
    Returns: beta, start_beta, end_beta (max_iters is set separately)
    """
    beta = v[0]
    end_beta = v[1]
    return beta, beta, end_beta  # start_beta = beta


def exponential_alpha_parser(v):
    """
    Parse optimization vector for ExponentialScheduler with alpha tuning.
    v[0]: beta (start_beta)
    v[1]: alpha
    Returns: beta, start_beta, alpha
    """
    beta = v[0]
    alpha = v[1]
    return beta, beta, alpha  # start_beta = beta