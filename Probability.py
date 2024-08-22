def calculateCondicionalProbability_obs(df, query_1, query_2):
    """
    Receives:
        df - pandas df where the probability will be calculated
        query_1 = represents the query that filters the given
        query_2 = represents the query of interest
    """
    upper = len(df.loc[query_1 & query_2])
    lower = len(df.loc[query_1])

    prob = upper / lower
    print(f"Probability: {prob}")
    return prob


def calcNormProbsbility(data, value, direction="lower"):
    """""" """""
    Recevies:
        data - pandas series or list 
        value - threshold calculate probability 
        direction - direction of the probability 
            if higher then the goal is to calculate the probability of a value being higher than threshold
            if lower then the goal is to calculate the probability of a value being lower than threshold
    
    """ """""" ""

    from scipy.stats import norm
    import numpy as np

    mean = np.mean(data)
    std_ = np.std(data)

    print("Sample statistics")
    print(f"Sample mean: {mean}")
    print(f"Sample std: {std_}")
    print(" ")

    if direction == "lower":
        prob = norm(mean, std_).cdf(value)
        print(f"Probability of being lower than {value}")
    elif direction == "upper":
        prob = 1 - norm(mean, std_).cdf(value)
        print(f"Probability of being higher than {value}")

    print(f"Probability: {prob}")

    return prob


def calcBinomialProbsbility(
    n_trials, probability_success, trials_target, direction="lower"
):
    """""" """""
    Recevies:
        n_trial - number of trials we want to test 
        probability_success - probability of event happening at each trial 
        trials_target - number of times event must occur 
        direction - represents the direction in which we want to calculate probability
        
    """ """""" ""

    from scipy.stats import binom
    import numpy as np

    print("Sample statistics")
    print(f"Number of trials: {n_trials}")
    print(f"Probability of sucess at each trial: {probability_success*100}%")
    print(f"Nuber of trials we want to see succeded: {trials_target}")
    print(" ")

    if direction == "lower":
        prob = binom(n=n_trials, p=probability_success).cdf(trials_target)
        print(
            f"Probability of event occuring less than {trials_target} times in a total of {n_trials} events"
        )
    elif direction == "upper":
        prob = 1 - binom(n=n_trials, p=probability_success).cdf(trials_target)
        print(
            f"Probability of event occuring more than {trials_target} times in a total of {n_trials} events"
        )

    elif direction == "exact":
        prob = binom(
            n=n_trials,
            p=probability_success,
        ).pmf(trials_target)
        print(
            f"Probability of event occuring exactly {trials_target} times in a total of {n_trials} events"
        )

    print(f"Probability: {prob*100}%")

    return prob


def calcPoissonProbability(lam, k, direction="lower"):
    """
    Receives:
        lam - lambda, the average rate of success (or event occurrence)
        k - number of successes (or events) we want to test
        direction - 'lower', 'upper', or 'exact' to calculate cumulative probability
    """
    from scipy.stats import poisson

    print("Assuming time is in minutes")
    print("Sample statistics")
    print(f"Lambda (average time between events): {lam}")
    print(f"Time we want to test: {k}")
    print(" ")

    if direction == "lower":
        prob = poisson.cdf(k, mu=lam)
        print(
            f"Probability of event taking less than or equal to {k} minutes to repeat"
        )
    elif direction == "upper":
        prob = 1 - poisson.cdf(k, mu=lam)
        print(f"Probability of event taking more than {k} to repeat")
    elif direction == "exact":
        prob = poisson.pmf(k, mu=lam)
        print(f"Probability of event occurring exactly in exactly {k}")

    print(f"Probability: {prob*100}%")

    return prob
