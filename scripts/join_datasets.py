import pandas as pd
import os
import sys
import random 

#  Clases 
#  Youtube -> 0
#  Twitch -> 1
#  Prime -> 2
#  Tiktok -> 3
#  Navegacion web -> 4

root_dir = sys.argv[1]
df_total = pd.DataFrame()

# Paso 1: Recolectar todos los archivos .csv
csv_files = []

for subdir, dir, files in os.walk(root_dir):
    for file in files:        
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename != 'total_dataset.csv' and filename != 'prime_yt_twitch.csv':
            csv_files.append(os.path.join(subdir, file))

random.shuffle(csv_files)

for file in csv_files:        
    print(file)
    df = pd.read_csv(file)

    ##TODO arreglar esto
    subdir = os.path.dirname(file)
    if subdir.endswith('youtube'):                
        df['class'] = 0  
    elif subdir.endswith('twitch'):
        df['class'] = 1
    elif subdir.endswith('prime'): 
        df['class'] = 2
    elif subdir.endswith('tiktok'): 
        df['class'] = 3
    elif subdir.endswith('navegacionweb'): 
        df['class'] = 4
    
            
    df_total = pd.concat([df_total, df], ignore_index=True)
    #df.to_csv(file, index=False)            

# Guardar el dataset combinado
df_total.to_csv(root_dir+'total_dataset.csv', index=False)
