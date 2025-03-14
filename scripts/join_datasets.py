import pandas as pd
import os
import sys

#  Clases 
#  Youtube -> 0
#  Twitch -> 1
#  Prime -> 2
#  Tiktok -> 3
#  Navegacion web -> 4

root_dir = sys.argv[1]
df_total = pd.DataFrame()


for subdir, dir, files in os.walk(root_dir):
    for file in files:        
        filename = os.fsdecode(file)

        if filename.endswith(".csv") and filename != 'total_dataset.csv':
            print(os.path.join(subdir, file))
            df = pd.read_csv(os.path.join(subdir, file))

            ##TODO arreglar esto
            print(subdir)
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
            df.to_csv(os.path.join(subdir, file), index=False)            

# Guardar el dataset combinado
df_total.to_csv(root_dir+'total_dataset.csv', index=False)
