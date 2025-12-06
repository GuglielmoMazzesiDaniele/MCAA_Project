from Configuration import steps_amount, schedule_type, T0, T_Min

def schedule_linear(t):
    return T0 - (T0 - T_Min) * (t / steps_amount)

def schedule_exp(t):
    r = (T_Min / T0) ** (1.0 / steps_amount)
    return T0 * (r ** t)

def schedule_power(t, tau=2_000, alpha=0.7):
    return T0 / ((1.0 + t / tau) ** alpha)

def schedule_two_phase(t, frac_warmup=0.3):
    warmup_steps = int(frac_warmup * steps_amount)
    if t < warmup_steps:
        return T0
    # Exponential from T0 to Tmin over the remaining steps
    remaining = steps_amount - warmup_steps
    k = t - warmup_steps
    r = (T_Min / T0) ** (1.0 / remaining)
    return T0 * (r ** k)

def temperature_schedule(t):
    if schedule_type == "linear":
        return schedule_linear(t)
    elif schedule_type == "exp":
        return schedule_exp(t)
    elif schedule_type == "power":
        return schedule_power(t, tau=5_000, alpha=0.7)
    elif schedule_type == "two_phase":
        return schedule_two_phase(t, frac_warmup=0.3)
    else:
        raise ValueError("Unknown schedule")