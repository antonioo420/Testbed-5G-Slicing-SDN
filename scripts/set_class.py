import sys
import os
import pandas as pd

#  Clases 
#  Youtube -> 0
#  Twitch -> 1
#  Prime -> 2
#  Tiktok -> 3
#  Navegacion web -> 4

if __name__ == "__main__":
    input_file = sys.argv[1]
    class_ = sys.argv[2]

    dataset = pd.read_csv(input_file)
    dataset['class'] = class_

    #file_name = input_file[:-4]
    dataset.to_csv(input_file, index=False) 
    