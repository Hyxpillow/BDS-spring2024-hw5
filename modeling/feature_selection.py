import pandas as pd
import math

def do_feature_selection():
    # information_gain
    raw_df = pd.read_csv("./Breast_Cancer_dataset.csv")
    # Calculate H(Status)
    p_alive = raw_df["Status"].value_counts(normalize=True)['Alive']
    p_dead = raw_df["Status"].value_counts(normalize=True)['Dead']
    entropy_status = -(p_alive * math.log2(p_alive) + p_dead * math.log2(p_dead))
    IG = {}
    # Calculate IG(*, Status) = H(Status) - conditional entropy
    for col_name in raw_df.columns:
        col = raw_df[col_name]
        # get the set of probabilities of values
        p_set = col.value_counts(normalize=True)
        condition_entropy_sum = 0
        for val in p_set.keys():
            col_conditional = raw_df[col == val]["Status"]
            p_set_conditional = col_conditional.value_counts(normalize=True)
            entropy_conditional = 0
            if 'Alive' in p_set_conditional.keys():
                p_alive_conditional = p_set_conditional['Alive']
                entropy_conditional += p_alive_conditional * math.log2(p_alive_conditional)
            if 'Dead' in p_set_conditional.keys():
                p_dead_conditional = p_set_conditional['Dead']
                entropy_conditional += p_dead_conditional * math.log2(p_dead_conditional)
            entropy_conditional *= -1
            condition_entropy_sum += p_set[val] * entropy_conditional
        IG[col_name] = entropy_status - condition_entropy_sum
    res = sorted(IG.items(), key=lambda x:x[1], reverse=True)
    for feature, ig in res:
        print("IG(%s) = %.4f" % (feature, ig))


    
