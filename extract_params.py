import json
import glob
import re

files = glob.glob("test_3_MNIST_edmd_dmps_*.ipynb")
files.sort()

# Sort files naturally (0, 1, 2... 10)
# from natsort import natsorted
# files = natsorted(files)
try:
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
except:
    pass


for fpath in files:
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        latent_dim = None
        num_epochs = None
        max_dm_samples = None
        autoencoder_type = None

        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = "".join(cell['source'])
                
                if latent_dim is None:
                    m = re.search(r'^latent_dim\s*=\s*(\d+)', source, re.MULTILINE)
                    if m: latent_dim = m.group(1)
                
                if num_epochs is None:
                    m = re.search(r'^num_epochs\s*=\s*(\d+)', source, re.MULTILINE)
                    if m: num_epochs = m.group(1)
                
                if max_dm_samples is None:
                    m = re.search(r'^max_dm_samples\s*=\s*(\d+)', source, re.MULTILINE)
                    if m: max_dm_samples = m.group(1)

                if autoencoder_type is None:
                    if "autoencoder = CNNAutoencoder" in source:
                        autoencoder_type = "CNN"
                    elif "autoencoder = MLPAutoencoder" in source:
                        autoencoder_type = "MLP"
        
        print(f"{fpath}: latent_dim={latent_dim}, num_epochs={num_epochs}, max_dm_samples={max_dm_samples}, type={autoencoder_type}")
    except Exception as e:
        print(f"Error reading {fpath}: {e}")
