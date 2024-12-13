import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

output_dir = './'  
filename = 'forested'

tif_files = [os.path.join(f'./{filename}', tif) for tif in os.listdir(f'./{filename}')]

resampled_data = []
for tif_file in tif_files:
    with rasterio.open(tif_file) as src:
        data = src.read()
        resampled_data.append(data)


stacked_data = np.concatenate(resampled_data, axis=0)  
num_bands, height, width = stacked_data.shape
reshaped_data = stacked_data.reshape(num_bands, height * width).T  

non_zero_mask = ~np.all(reshaped_data == 0, axis=1)
filtered_data = reshaped_data[non_zero_mask]
print(f'reshaped data: {reshaped_data.shape}, filter data: {filtered_data.shape}')

pca = PCA()
pca.fit(filtered_data)
pca_components = pca.components_  
band_contribution = np.abs(pca_components)


band_names = [tif_file.split('/')[-1].split('_')[-1].split('.')[0] for tif_file in tif_files]
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
num_components = np.argmax(cumulative_variance >= 0.95) + 1 
print(f'components to keep: {num_components}')

def create_pca_visualization(pca_components, explained_variance, band_names, output_dir):
    """
    PCA visualization showing variance explained and band contributions
    """
    
    band_contribution = np.abs(pca_components)
    normalized_contributions = np.zeros_like(band_contribution)
    for i in range(band_contribution.shape[0]):
        normalized_contributions[i] = band_contribution[i] * explained_variance[i] / band_contribution[i].sum()
    
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(explained_variance))
    colors = plt.cm.tab20(np.linspace(0, 1, len(band_names)))
    
    for band_idx, band_name in enumerate(band_names):
        contributions = normalized_contributions[:, band_idx]
        plt.bar(range(1, len(explained_variance) + 1), contributions, 
               bottom=bottom, label=f'{band_name}',
               color=colors[band_idx])
        bottom += contributions
        print(band_name, list(range(1, len(explained_variance) + 1)), contributions)
    
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Explained')
    plt.title('PCA: Variance Explained and Band Contributions')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.ylim(0, max(explained_variance) * 1.1)  
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    
    cumulative_variance = np.cumsum(explained_variance)
    for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
        plt.text(i + 1, var, f'{cum_var:.1%}', 
                ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'{filename}_temp.png'), 
                bbox_inches='tight', dpi=300, format='png')
    plt.close()

create_pca_visualization(
    pca_components=pca.components_,
    explained_variance=pca.explained_variance_ratio_,
    band_names=band_names,
    output_dir=output_dir
)



