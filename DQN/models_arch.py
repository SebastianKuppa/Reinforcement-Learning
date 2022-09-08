import constants

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