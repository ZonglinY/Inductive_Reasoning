import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi


def is_nan(d):
    if isinstance(d, str) == True:
        return False
    elif np.isnan(d) == True:
        return True
    else:
        print("d: ", d)
        return False


def validator(data_type):
    assert data_type == "train" or data_type == "test"
    data = pd.read_excel("Hypothetical_Induction_{}.xlsx".format(data_type))
    len_d = len(data)
    print("len_d: ", len_d)

    for cur_id in range(len_d):
        print("cur_id: ", cur_id)
        l11, l12, l21, l22, l31, l32, s11, s12, s21, s22, s31, s32 = data['fact 1.1'][cur_id], data['fact 1.2'][cur_id], data['fact 2.1'][cur_id], data['fact 2.2'][cur_id], data['fact 3.1'][cur_id], data['fact 3.2'][cur_id], data['short fact 1.1'][cur_id], data['short fact 1.2'][cur_id], data['short fact 2.1'][cur_id], data['short fact 2.2'][cur_id], data['short fact 3.1'][cur_id], data['short fact 3.2'][cur_id]
        # no empty facts
        assert (is_nan(l11) and is_nan(l12)) == False
        assert (is_nan(l21) and is_nan(l22)) == False
        assert (is_nan(l31) and is_nan(l32)) == False
        assert (is_nan(s11) and is_nan(s12)) == False
        assert (is_nan(s21) and is_nan(s22)) == False
        assert (is_nan(s31) and is_nan(s32)) == False
        # most of the short facts should be contained in corresponding long facts
        def check_short_in_long(long, short):
            if is_nan(long):
                return True
            elif is_nan(short):
                print("Warning: long exists but short is nan")
                return True
            elif short in long:
                print("Perfect!")
                return True
            else:
                long = long.replace(u'\xa0', u' ').strip()
                short = short.replace(u'\xa0', u' ').strip()
                bm25 = BM25Okapi([long.split(" ")])
                short_tokenized = short.split(" ")
                simi_scores = bm25.get_scores(short_tokenized)
                # print("simi_scores: ", simi_scores)
                if abs(simi_scores) < 0.5:
                    print("Warning: long: {};\nshort:{}".format(long, short))
                    return False
                elif abs(simi_scores) < 3:
                    print("Warning: long: {};\nshort:{}".format(long, short))
                    return True
                else:
                    return True

        vali =  check_short_in_long(l11, s11)
        assert vali == True
        vali =  check_short_in_long(l12, s12)
        assert vali == True
        vali =  check_short_in_long(l21, s21)
        assert vali == True
        vali =  check_short_in_long(l22, s22)
        assert vali == True
        vali =  check_short_in_long(l31, s31)
        assert vali == True
        vali =  check_short_in_long(l32, s32)
        assert vali == True


def main():
    # "test"
    # data_type = "train"
    validator("train")
    validator("test")



















if __name__ == "__main__":
    main()
