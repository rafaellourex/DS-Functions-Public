import pandas as pd
import numpy as np


def T_Test_ind(s1, s2, type_="two-sided"):
    """""" """""" """'
    This is a test for the null hypothesis that two related or repeated samples have identical average (expected) values.
    Receives:
        s1, as 1st population
        s2 as second population 
        type_ as type of test ('two-sided','greater','less')
        
    Conditions:
        s1 and s2 are dependent
        
    Performes a t-test for 2 dependent populations 
        If type_ == 'two-sided' than the H0 will be that both populations will have the same mean 
        If type_ == 'greater' or 'less' than H1 will be that the mean of s1 is greater or smaller than s2
    
    """ """""" """""" ""

    from scipy import stats

    print("Performing T Test")
    print(f"Type of test: {type_}")
    t_stat, p_value = stats.ttest_ind(
        s1,
        s2,
        alternative=type_,
    )

    if p_value < 0.05:
        print("We reject the null hypothesis with a level of confidence of 95%")
        #         print('Portfolio returns are greater than randomly generated portfolio returns.')
        print(f"P-Value: {p_value}")

    else:
        print("We dont reject H0 with a level of confidence of 95%")
        #         print('Portfolio returns arent greater than randomly generated portfolio returns' )
        print(f"P-Value: {p_value}")

    return (t_stat, p_value)


## 2 Sample Tests
def T_Test(s1, s2, type_="two-sided"):
    """""" """""" """'
    This is a test for the null hypothesis that two related or repeated samples have identical average (expected) values.
    Receives:
        s1, as 1st population
        s2 as second population 
        type_ as type of test ('two-sided','greater','less')
        
    Conditions:
        s1 and s2 are dependent
        
    Performes a t-test for 2 dependent populations 
        If type_ == 'two-sided' than the H0 will be that both populations will have the same mean 
        If type_ == 'greater' or 'less' than H1 will be that the mean of s1 is greater or smaller than s2
    
    """ """""" """""" ""

    from scipy import stats

    print("Performing T Test")
    print(f"Type of test: {type_}")
    t_stat, p_value = stats.ttest_rel(
        s1,
        s2,
        alternative=type_,
    )

    if p_value < 0.05:
        print("We reject the null hypothesis with a level of confidence of 95%")
        #         print('Portfolio returns are greater than randomly generated portfolio returns.')
        print(f"P-Value: {p_value}")

    else:
        print("We dont reject H0 with a level of confidence of 95%")
        #         print('Portfolio returns arent greater than randomly generated portfolio returns' )
        print(f"P-Value: {p_value}")

    return (t_stat, p_value)


def T1_Test(s1, popmean, type_="two-sided"):
    """""" """""" """'
    This is a test for the null hypothesis that two related or repeated samples have identical average (expected) values.
    Receives:
        s1, as 1st population
        population mean - value we want to test
        type_ as type of test ('two-sided','greater','less')
        
    Conditions:
        s1 and s2 are dependent
        
    Performes a t-test for 2 dependent populations 
        If type_ == 'two-sided' than the H0 will be that both populations will have the same mean 
        If type_ == 'greater' or 'less' than H1 will be that the mean of s1 is greater or smaller than s2
    
    """ """""" """""" ""

    from scipy import stats

    print("Performing T Test")
    print(f"Type of test: {type_}")
    t_stat, p_value = stats.ttest_1samp(a=s1, popmean=popmean, alternative="two-sided")

    if p_value < 0.05:
        print("We reject the null hypothesis with a level of confidence of 95%")
        #         print('Portfolio returns are greater than randomly generated portfolio returns.')
        print(f"P-Value: {p_value}")

    else:
        print("We dont reject H0 with a level of confidence of 95%")
        #         print('Portfolio returns arent greater than randomly generated portfolio returns' )
        print(f"P-Value: {p_value}")

    return (t_stat, p_value)


def Z_Test(s1, s2, type_="two-sided"):
    """""" """""" """'
    This is a test for the null hypothesis that two related or repeated samples have identical average (expected) values.
    
    Receives:
        s1, as 1st population
        s2 as second population 
        type_ as type of test ('two-sided','greater','less')
    
    Conditions:
        s1 and s2 are independent 
        lenght of s1 and s2 > 30
    
        
    Performes a Z-test for 2 independent populations 
        If type_ == 'two-sided' than the H0 will be that both populations will have the same mean 
        If type_ == 'greater' or 'less' than H1 will be that the mean of s1 is greater or smaller than s2
    
    """ """""" """""" ""

    from scipy import stats
    from statsmodels.stats import weightstats as stest

    print("Performing Z Test")
    t_stat, p_value = stest.ztest(
        s1,
        s2,
        alternative=type_,
    )

    if p_value < 0.05:
        print("We reject the null hypothesis with a level of confidence of 95%")
        print(f"P-Value: {p_value}")

    else:
        print("We dont reject H0 with a level of confidence of 95%")
        print(f"P-Value: {p_value}")

    return (t_stat, p_value)


def wilcox_test(s1, s2, type_="two-sided"):
    """""" """""" """'
    The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same 
    distribution. In particular, it tests whether the distribution of 
    the differences x - y is symmetric about zero. 
    It is a non-parametric version of the paired T-test
    
    
    Receives:
        s1, as 1st population
        s2 as second population 
        type_ as type of test ('two-sided','greater','less')
        
        Note: Di = Yi - Xi
        
    Conditions:
        S1 and S2 are dependent
        S1 and S2 must have the same lenght
        Di are independent
    
    Performes a t-test for 2 dependent populations 
        If type_ == 'two-sided' than the H0 will be that D is symmetric around 0
        If type_ == 'greater' or 'less' than H1 will be that the D is greater or smaller than 0
    
    """ """""" """""" ""

    from scipy import stats

    print(f"Performing {type_} Wilcox Test.")
    print(" ")
    t_stat, p_value = stats.wilcoxon(
        s1,
        y=s2,
        zero_method="wilcox",
        correction=False,
        alternative=type_,
    )

    if p_value < 0.05:
        print("We reject the null hypothesis with a level of confidence of 95%")
        #         print('Portfolio returns are greater than randomly generated portfolio returns.')
        print(f"P-Value: {p_value}")

    else:
        print("We dont reject H0 with a level of confidence of 95%")
        #         print('Portfolio returns arent greater than randomly generated portfolio returns' )
        print(f"P-Value: {p_value}")

    return t_stat, p_value


# wilcox_test(s1,s2, 'greater')


#### Mann-Whitney Test
def mann_whitney_test(s1, s2, type_="two-sided"):
    """""" """""" """'
    The Mann-Whitney U test is a nonparametric test of the null hypothesis that the distribution 
    underlying sample x is the same as the distribution underlying sample y. 
    It is often used as a test of difference in location between distributions.
    
    
    Receives:
        s1, as 1st population
        s2 as second population 
        type_ as type of test ('two-sided','greater','less')
        
        Note: Di = Yi - Xi
        
    Conditions:
        S1 and S2 are indedependent
        S1 and S2 can have different lenghts
        
    
    Performes a t-test for 2 dependent populations 
        If type_ == 'two-sided' than the H0 will be that D is symmetric around 0
        If type_ == 'greater' or 'less' than H1 will be that the D is greater or smaller than 0
    
    """ """""" """""" ""
    import scipy as sc
    from scipy import stats

    print(f"Performing {type_} Mann-Whitney Test.")
    print(" ")
    t_stat, p_value = stats.mannwhitneyu(
        s1,
        y=s2,
        alternative="greater",
    )

    if p_value < 0.05:
        print("We reject the null hypothesis with a level of confidence of 95%")
        #         print('Portfolio returns are greater than randomly generated portfolio returns.')
        print(f"P-Value: {p_value}")

    else:
        print("We dont reject H0 with a level of confidence of 95%")
        #         print('Portfolio returns arent greater than randomly generated portfolio returns' )
        print(f"P-Value: {p_value}")

    t_stat, p_value


# man_whitney_test(s1,s2, 'greater')


# variance test


def levene_test(s1, s2, type_="two-sided"):
    """""" """""" """'
    The Levene test tests the null hypothesis that all input samples are from populations with equal variances.
    Levene’s test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations 
a4    from normality.
    
    
    Receives:
        s1, as 1st population
        s2 as second population 
        type_ as type of test ('two-sided','greater','less')
        
    Conditions:
        S1 and S2 are indedependent
        S1 and S2 can have different lenghts
        
    
    Performes a t-test for 2 dependent populations 
    
    """ """""" """""" ""
    import scipy as sc
    from scipy import stats

    print(f"Performing Levene Test for variance.")
    print(" ")
    t_stat, p_value = stats.levene(s1, s2)

    if p_value < 0.05:
        print("We reject the null hypothesis with a level of confidence of 95%")
        print("At least one of the populations doesnt have the same variance ")
        print(f"P-Value: {p_value}")

    else:
        print("We dont reject H0 with a level of confidence of 95%")

        print("Populations have all the same variance.")
        print(f"P-Value: {p_value}")


# levene_test(s1,s2,)


def barlett_test(s1, s2, type_="two-sided"):
    """""" """""" """'
    The Levene test tests the null hypothesis that all input samples are from populations with equal variances. 
    Levene’s test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations 
    from normality.
    
    Receives:
        s1, as 1st population
        s2 as second population 
        type_ as type of test ('two-sided','greater','less')
        
        
    Conditions:
        S1 and S2 are indedependent
        S1 and S2 can have different lenghts
        
    
    Performes a t-test for 2 dependent populations 
    
    """ """""" """""" ""
    import scipy as sc
    from scipy import stats

    print(f"Performing Barlett Test for variance.")
    print(" ")
    t_stat, p_value = stats.bartlett(s1, s2)

    if p_value < 0.05:
        print("We reject the null hypothesis with a level of confidence of 95%")
        print("At least one of the populations doesnt have the same variance ")
        print(f"P-Value: {p_value}")

    else:
        print("We dont reject H0 with a level of confidence of 95%")

        print("Populations have all the sane variance.")
        print(f"P-Value: {p_value}")


# barlett_test(s1,s2,)


def kruskalWallis_test(
    s1,
    s2,
):
    """""" """""" """'
    The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal. 
    It is a non-parametric version of ANOVA. The test works on 2 or more independent samples, which may have different sizes. 
    Note that rejecting the null hypothesis does not indicate which of the groups differs. 
    Post hoc comparisons between groups are required to determine which groups are different.
    
    Receives:
        s1, as 1st population
        s2 as second population 
        
    Conditions:
        S1 and S2 are indedependent
        S1 and S2 can have different lenghts
        
    
    
    """ """""" """""" ""
    import scipy as sc
    from scipy import stats

    print(f"Performing Kruskal Wallis Test.")
    print(" ")
    t_stat, p_value = stats.kruskal(s1, s2)

    if p_value < 0.05:
        print("We reject the null hypothesis with a level of confidence of 95%")
        print("At least one of the populations doesnt have the same distribution")
        print(f"P-Value: {p_value}")

    else:
        print("We dont reject H0 with a level of confidence of 95%")

        print("Populations have all the same distribution.")
        print(f"P-Value: {p_value}")


# kruskalWallis_test(s1,s2,)


def friedman_test(s1, s2, s3):
    """""" """""" """'
    The Friedman test tests the null hypothesis that repeated samples of the same individuals have the same distribution. 
    It is often used to test for consistency among samples obtained in different ways. 
    For example, if two sampling techniques are used on the same set of individuals, 
    the Friedman test can be used to determine if the two sampling techniques are consistent.
    
    Receives:
        s1, as 1st population
        s2 as second population 
        
    Conditions:
        S1 and S2 are indedependent
        S1 and S2 have the same lenghts
        s1 and s2 are ordered in a manner of interest (eg: by date)
        
    
    Performes a t-test for 2 dependent populations 
    
    """ """""" """""" ""
    import scipy as sc
    from scipy import stats

    print(f"Performing Friedman test.")
    print(" ")
    t_stat, p_value = stats.friedmanchisquare(s1, s2, s3)

    if p_value < 0.05:
        print("We reject the null hypothesis with a level of confidence of 95%")
        print("At least one of the populations doesnt have the same distribution")
        print(f"P-Value: {p_value}")

    else:
        print("We dont reject H0 with a level of confidence of 95%")

        print("Populations have all the same distribution.")
        print(f"P-Value: {p_value}")


# friedman_test(s1,s2,s3)


def chiSquare_Prob_test(df_contigency):
    """""" """""" """'
    This function computes the chi-square statistic and p-value for the hypothesis test of independence of the observed 
    frequencies in the contingency table [1] observed. 
    The expected frequencies are computed based on the marginal sums under the assumption of independence; 
    see scipy.stats.contingency.expected_freq. 
    The number of degrees of freedom is (expressed using numpy functions and attributes):
    
    Receives:
        rXm contigency table  
        
    Conditions:
        Samples are mutually independent
        Each observation is categorized by 1 category given 2 possible categories
    
    """ """""" """""" ""
    import scipy as sc
    from scipy import stats

    print(f"Performing Chi-Squared Contigency test.")
    print(" ")
    t_stat, p_value, dof, expected = stats.chi2_contingency(
        df_contigency,
    )

    if p_value < 0.05:
        print("We reject the null hypothesis with a level of confidence of 95%")
        print("The categories dont have the same probability of occuring")
        print(f"P-Value: {p_value}")

    else:
        print("We dont reject H0 with a level of confidence of 95%")
        print("The categories have the same probability of occuring")
        print(f"P-Value: {p_value}")
    return expected


# chiSquare_Prob_test(df)


# correlation tests
def SpearmanDependence_Test(s1, s2):
    """""" """""" """'
    This function computes the chi-square statistic and p-value for the hypothesis test of independence of the observed 
    frequencies in the contingency table [1] observed. 
    The expected frequencies are computed based on the marginal sums under the assumption of independence; 
    see scipy.stats.contingency.expected_freq. 
    The number of degrees of freedom is (expressed using numpy functions and attributes):
    
    Receives:
        rXm contigency table  
        
    Conditions:
        random samples
        Scale is at least ordinal
    
    """ """""" """""" ""

    import scipy

    print("Performing Spearman correlation test")
    corr, p_value = scipy.stats.spearmanr(s1, s2)
    print(f"Correlation: {corr}")

    if p_value < 0.05:
        print("We reject the null hypothesis with a level of confidence of 95%")
        print("Features are dependent from each other.")
        print(f"P-Value: {p_value}")

    else:
        print("We dont reject H0 with a level of confidence of 95%")
        print("Features are indedependent from each other")
        print(f"P-Value: {p_value}")


# SpearmanDependence_Test(s1,s2)


def BootstrapSampling(data, pct, iterations):
    """""" """"
    Receives:
        data - pandas df containing the data we want to sample 
        pct - 5% of the total data we want to use in each bootstrap
        iteration - nr of repetitions 
         
    """ """""" ""

    pct = pct
    iterations = iterations
    size = int(len(data) * pct)

    full_boot = pd.DataFrame()

    metrics_list = []
    for iter in np.arange(0, iterations):
        bootstrap = data.sample(n=size, replace=True)
        boot_avg = bootstrap.mean()
        boot_std = bootstrap.std()
        bootstrap["iteration"] = iter

        metrics_dict = dict()
        metrics_dict["bootStrap_mean"] = boot_avg
        metrics_dict["bootStrap_std"] = boot_std
        metrics_list.append(metrics_dict)
        full_boot = pd.concat([full_boot, bootstrap], axis=0)

    metrics_df = pd.DataFrame.from_records(metrics_list)

    ic_90 = np.percentile(metrics_df["bootStrap_mean"], [10, 90])

    ic_95 = np.percentile(metrics_df["bootStrap_mean"], [5, 95])

    ic_99 = np.percentile(metrics_df["bootStrap_mean"], [1, 99])

    ic_90_std = np.percentile(metrics_df["bootStrap_std"], [10, 90])

    ic_95_std = np.percentile(metrics_df["bootStrap_std"], [5, 95])

    ic_99_std = np.percentile(metrics_df["bootStrap_std"], [1, 99])

    print("Confidence Intervals of the mean:")
    print(f"90% : {ic_90}")
    print(f"95% : {ic_95}")
    print(f"99% : {ic_99}")

    print(" ")
    print("Confidence Intervals of the standard deviation:")
    print(f"90% : {ic_90_std}")
    print(f"95% : {ic_95_std}")
    print(f"99% : {ic_99_std}")

    return (full_boot, metrics_df)
