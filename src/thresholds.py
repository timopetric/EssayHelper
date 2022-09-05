from helpers import Thresholds

thresholds_number = Thresholds(
    # only number
    number32=0.056,
    case42=1,
    
    skip_if_first_sugg_below=0, direct_replacement_if_same_lemma=1,
    number12=1, number13=1, number21=1, number23=1, number31=1, case24=1, case53=1, 
    case35=1, case12=1, case16=1, case23=1, case41=1, case45=1, case64=1, case51=1,
    case61=1, case54=1, case65=1, case32=1, case36=1, case13=1, case26=1, case63=1,
    case56=1, case52=1, case34=1, case46=1, case15=1, case25=1, case14=1, case21=1,
    case43=1, case31=1, case62=1)

thresholds_number_mcd_off = Thresholds(
    # only number
    number32=0.032,
    case42=1,
    
    skip_if_first_sugg_below=0, direct_replacement_if_same_lemma=1,
    number12=1, number13=1, number21=1, number23=1, number31=1, case24=1, case53=1, 
    case35=1, case12=1, case16=1, case23=1, case41=1, case45=1, case64=1, case51=1,
    case61=1, case54=1, case65=1, case32=1, case36=1, case13=1, case26=1, case63=1,
    case56=1, case52=1, case34=1, case46=1, case15=1, case25=1, case14=1, case21=1,
    case43=1, case31=1, case62=1)

thresholds_case = Thresholds(
    # only case
    case42=0.0097,
    number32=1,

    skip_if_first_sugg_below=0, direct_replacement_if_same_lemma=1,
    number12=1, number13=1, number21=1, number23=1, number31=1, case24=1, case53=1, 
    case35=1, case12=1, case16=1, case23=1, case41=1, case45=1, case64=1, case51=1,
    case61=1, case54=1, case65=1, case32=1, case36=1, case13=1, case26=1, case63=1,
    case56=1, case52=1, case34=1, case46=1, case15=1, case25=1, case14=1, case21=1,
    case43=1, case31=1, case62=1)

thresholds_case_mcd_off = Thresholds(
    # only case
    case42=0.007,
    number32=1,

    skip_if_first_sugg_below=0, direct_replacement_if_same_lemma=1,
    number12=1, number13=1, number21=1, number23=1, number31=1, case24=1, case53=1, 
    case35=1, case12=1, case16=1, case23=1, case41=1, case45=1, case64=1, case51=1,
    case61=1, case54=1, case65=1, case32=1, case36=1, case13=1, case26=1, case63=1,
    case56=1, case52=1, case34=1, case46=1, case15=1, case25=1, case14=1, case21=1,
    case43=1, case31=1, case62=1)

thresholds_both = Thresholds(
    # both case and number
    case42=0.006,
    number32=0.03,

    skip_if_first_sugg_below=0, direct_replacement_if_same_lemma=1,
    number12=1, number13=1, number21=1, number23=1, number31=1, case24=1, case53=1, 
    case35=1, case12=1, case16=1, case23=1, case41=1, case45=1, case64=1, case51=1,
    case61=1, case54=1, case65=1, case32=1, case36=1, case13=1, case26=1, case63=1,
    case56=1, case52=1, case34=1, case46=1, case15=1, case25=1, case14=1, case21=1,
    case43=1, case31=1, case62=1)

thresholds_both_mcd_off = Thresholds(
    # both case and number
    case42=0.0013,
    number32=0.036,

    skip_if_first_sugg_below=0, direct_replacement_if_same_lemma=1,
    number12=1, number13=1, number21=1, number23=1, number31=1, case24=1, case53=1, 
    case35=1, case12=1, case16=1, case23=1, case41=1, case45=1, case64=1, case51=1,
    case61=1, case54=1, case65=1, case32=1, case36=1, case13=1, case26=1, case63=1,
    case56=1, case52=1, case34=1, case46=1, case15=1, case25=1, case14=1, case21=1,
    case43=1, case31=1, case62=1)


def get_thresholds(thresholds_option: int, mcd_on: bool):
    assert thresholds_option in {0,1,2}, "Thresholds option should be one of: 0 = both, 1 = case, 2 = number"
    assert isinstance(mcd_on, bool), "mcd_on param should be True or False"
    
    print(f"Using thresholds options: {thresholds_option}, mcd_on: {mcd_on}.")

    if thresholds_option == 0:
        return thresholds_both if mcd_on else thresholds_both_mcd_off
    elif thresholds_option == 1:
        return thresholds_case if mcd_on else thresholds_case_mcd_off
    elif thresholds_option == 2:
        return thresholds_number if mcd_on else thresholds_number_mcd_off
    else:
        raise ValueError("Error selecting threshold option.")