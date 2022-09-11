import constants
import pandas as pd

models_arch = [{"conv_list": [32], "dense_list": [32, 32], "util_list": ["ECC2", "1A-5Ac"],
                "MINIBATCH_SIZE": 128, "best_only": False,
                "EF_Settings": {"EF_Enabled": False}, "ECC_Settings": {"ECC_Enabled": False}},

               {"conv_list": [32], "dense_list": [32, 32, 32], "util_list": ["ECC2", "1A-5Ac"],
                "MINIBATCH_SIZE": 128, "best_only": False,
                "EF_Settings": {"EF_Enabled": False}, "ECC_Settings": {"ECC_Enabled": False}},

               {"conv_list": [32], "dense_list": [32, 32], "util_list": ["ECC2", "1A-5Ac"],
                "MINIBATCH_SIZE": 128, "best_only": False,
                "EF_Settings": {"EF_Enabled": True, "FLUCTUATIONS": 2},
                "ECC_Settings": {"ECC_Enabled": True, "MAX_EPS_NO_INC": int(constants.EPISODES * 0.2)}}]


# A dataframe used to store grid search results
res = pd.DataFrame(columns=["Model Name","Convolution Layers", "Dense Layers", "Batch Size", "ECC", "EF",
                            "Best Only" , "Average Reward", "Best Average", "Epsilon 4 Best Average",
                            "Best Average On", "Max Reward", "Epsilon 4 Max Reward", "Max Reward On",
                            "Total Training Time (min)", "Time Per Episode (sec)"])