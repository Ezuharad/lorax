import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


output_dir = './'  

target_height = 10980
target_width = 10980

tif_files = [os.path.join('./data', tif) for tif in os.listdir('./data')]

resampled_data = []

for tif_file in tif_files:
    with rasterio.open(tif_file) as src:
        
        data = src.read(
            out_shape=(src.count, target_height, target_width),
            resampling=Resampling.bilinear
        )
        resampled_data.append(data)


stacked_data = np.concatenate(resampled_data, axis=0)  


num_bands, height, width = stacked_data.shape
reshaped_data = stacked_data.reshape(num_bands, height * width).T  


pca = PCA()
pca.fit(reshaped_data)

pca_components = pca.components_  
band_contribution = np.abs(pca_components)

band_names = [tif_file.split('/')[-1].split('_')[-1].split('.')[0] for tif_file in tif_files]


explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
num_components = np.argmax(cumulative_variance >= 0.95) + 1 
print(f'components to keep: {num_components}')
x = np.arange(num_components)  

def create_pca_visualization(pca_components, explained_variance, band_names, output_dir):
    """
    Create a PCA visualization showing variance explained and band contributions
    """
    # Calculate absolute contributions of each band
    band_contribution = np.abs(pca_components)
    
    # Normalize band contributions so they sum to the explained variance for each PC
    normalized_contributions = np.zeros_like(band_contribution)
    for i in range(band_contribution.shape[0]):
        normalized_contributions[i] = band_contribution[i] * explained_variance[i] / band_contribution[i].sum()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot stacked bars for each principal component
    bottom = np.zeros(len(explained_variance))
    
    # Create color map for bands
    colors = plt.cm.tab20(np.linspace(0, 1, len(band_names)))
    
    for band_idx, band_name in enumerate(band_names):
        contributions = normalized_contributions[:, band_idx]
        plt.bar(range(1, len(explained_variance) + 1), contributions, 
               bottom=bottom, label=f'{band_name}',
               color=colors[band_idx])
        bottom += contributions
    
    # Customize the plot
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Explained')
    plt.title('PCA: Variance Explained and Band Contributions')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.ylim(0, max(explained_variance) * 1.1)  # Add 10% padding to y-axis
    
    # Add legend with two columns
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    
    # Add text annotations for cumulative variance
    cumulative_variance = np.cumsum(explained_variance)
    for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
        plt.text(i + 1, var, f'{cum_var:.1%}', 
                ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'tile5.png'), 
                bbox_inches='tight', dpi=300, format='png')
    plt.close()

create_pca_visualization(
    pca_components=pca.components_,
    explained_variance=pca.explained_variance_ratio_,
    band_names=band_names,
    output_dir=output_dir
)

