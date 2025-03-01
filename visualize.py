# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

import scipy
import scipy.stats
from imageio import imsave
import cv2


def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap='jet', smooth=True):
    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')  # ignore divide by zero err
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(activations, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb(im, cmap, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig


def compute_ratemaps(model, trajectory_generator, options, res=20, n_avg=None, Ng=512, idxs=None):
    '''Compute spatial firing fields'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ng])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Ng, res, res]) 
    counts  = np.zeros([res, res])

    for index in range(n_avg):
        inputs, pos_batch, _ = trajectory_generator.get_test_batch()
        g_batch = model.g(inputs).detach().cpu().numpy()
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        g_batch = g_batch[:,:,idxs].reshape(-1, Ng)
        
        g[index] = g_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res

        for i in range(options.batch_size*options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >=0 and x < res and y >=0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += g_batch[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]
                
    g = g.reshape([-1, Ng])
    pos = pos.reshape([-1, 2])

    # # scipy binned_statistic_2d is slightly slower
    # activations = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
    rate_map = activations.reshape(Ng, -1)

    return activations, rate_map, g, pos


def save_ratemaps(model, trajectory_generator, options, step, res=20, n_avg=None):
    if not n_avg:
        n_avg = 1000 // options.sequence_length
    activations, rate_map, g, pos = compute_ratemaps(model, trajectory_generator,
                                                     options, res=res, n_avg=n_avg)
    rm_fig = plot_ratemaps(activations, n_plots=len(activations))
    imdir = options.save_dir + "/" + options.run_ID
    imsave(imdir + "/" + str(step) + ".png", rm_fig)


def save_autocorr(sess, model, save_name, trajectory_generator, step, flags):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range=((-1.1, 1.1), (-1.1, 1.1))
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, coord_range, masks_parameters)
    
    res = dict()
    index_size = 100
    for _ in range(index_size):
      feed_dict = trajectory_generator.feed_dict(flags.box_width, flags.box_height)
      mb_res = sess.run({
          'pos_xy': model.target_pos,
          'bottleneck': model.g,
      }, feed_dict=feed_dict)
      res = utils.concat_dict(res, mb_res)
        
    filename = save_name + '/autocorrs_' + str(step) + '.pdf'
    imdir = flags.save_dir + '/'
    out = utils.get_scores_and_plot(
                latest_epoch_scorer, res['pos_xy'], res['bottleneck'],
                imdir, filename)


def compute_ae_ratemaps(model, ae, trajectory_generator, options, res=20, n_avg=None, Ng=512, Ns=512, idxs=None):
    '''Compute spatial firing fields for SAE latents'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not np.any(idxs):
        idxs = np.arange(Ns)
    idxs = idxs[:Ns]

    # Use Ns for SAE latent dimensions
    g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ns])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Ns, res, res]) 
    counts = np.zeros([res, res])

    for index in range(n_avg):
        inputs, pos_batch, _ = trajectory_generator.get_test_batch()
        # Get grid cell activations and encode with SAE
        g_batch = model.g(inputs)
        sae_batch = ae.encode(g_batch.reshape(-1, g_batch.shape[-1])).detach().cpu().numpy()
        sae_batch = sae_batch.reshape(g_batch.shape[0], g_batch.shape[1], -1)
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        sae_batch = sae_batch[:,:,idxs].reshape(-1, Ns)
        
        g[index] = sae_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res

        for i in range(options.batch_size*options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >=0 and x < res and y >=0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += sae_batch[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]
                
    g = g.reshape([-1, Ns])
    pos = pos.reshape([-1, 2])

    rate_map = activations.reshape(Ns, -1)

    return activations, rate_map, g, pos


def compute_pos_ratemaps(trajectory_generator, options, res=20, n_avg=None, Np=512, idxs=None):
    '''Compute spatial firing fields for place cells directly from trajectory generator outputs'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not np.any(idxs):
        idxs = np.arange(Np)
    idxs = idxs[:Np]

    p = np.zeros([n_avg, options.batch_size * options.sequence_length, Np])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Np, res, res]) 
    counts = np.zeros([res, res])

    for index in range(n_avg):
        inputs, pos_batch, pc_outputs = trajectory_generator.get_test_batch()
        # Use pc_outputs directly instead of getting from model
        p_batch = pc_outputs.detach().cpu().numpy()
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        p_batch = p_batch.reshape(-1, Np)[:, idxs]
        
        p[index] = p_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res

        for i in range(options.batch_size*options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >=0 and x < res and y >=0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += p_batch[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]
                
    p = p.reshape([-1, Np])
    pos = pos.reshape([-1, 2])

    rate_map = activations.reshape(Np, -1)

    return activations, rate_map, p, pos



# Perform UMAP visualization in 3D, with optional PCA preprocessing
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_umap_3d(data, use_pca_first=True, pca_components=6, random_state=42):
    """
    Visualize data in 3D using UMAP, with optional PCA preprocessing.
    
    Parameters:
    - data: The input data
    - use_pca_first: Whether to use PCA preprocessing
    - pca_components: Number of PCA components if PCA is used
    - random_state: Random seed for reproducibility
    """
    # Apply PCA preprocessing if requested
    if use_pca_first:
        print(f"Applying PCA to reduce dimensions from {data.shape[1]} to {pca_components}...")
        pca = PCA(n_components=pca_components)
        data_transformed = pca.fit_transform(data)
        explained_variance = sum(pca.explained_variance_ratio_) * 100
        print(f"Explained variance with {pca_components} components: {explained_variance:.2f}%")
    else:
        data_transformed = data
    
    # Apply UMAP for 3D visualization
    print("Applying UMAP to create 3D visualization...")
    reducer = umap.UMAP(
        n_components=3, 
        n_neighbors=15, # 15
        metric='cosine',
        min_dist=0.5, # 0.5
        init='spectral',
        random_state=random_state
    )
    embedding = reducer.fit_transform(data_transformed)
    
    # Visualize in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        embedding[:, 2],
        s=10,  # smaller point size for better visualization
        alpha=0.5,
        c=embedding[:, 2],  # color by 3rd dimension
        cmap='viridis'
    )
    
    # Add colorbar and labels
    plt.colorbar(scatter, ax=ax, pad=0.1)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_zlabel('UMAP Dimension 3')
    
    title = "3D UMAP Visualization of Grid Cell Activities"
    if use_pca_first:
        title += f" (with PCA preprocessing to {pca_components} dimensions)"
    ax.set_title(title)
    
    # Improve 3D visualization appearance
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True)
    
    plt.tight_layout()
    return fig, ax, embedding


# Interactive 3D plotting of UMAP results using Plotly
import plotly.express as px
import plotly.graph_objects as go

# Create interactive 3D plot from the existing embedding
def create_interactive_3d_plot(embedding, title="Interactive 3D UMAP Visualization"):
    """
    Create an interactive 3D plot using Plotly
    
    Parameters:
    - embedding: The 3D embedding data (n_samples, 3)
    - title: Plot title
    
    Returns:
    - fig: Plotly figure that can be displayed
    """
    # Create a dataframe for plotly
    import pandas as pd
    df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'UMAP3': embedding[:, 2],
        'Color': embedding[:, 2]  # Using 3rd dimension for color
    })
    
    # Create interactive 3D scatter plot
    fig = px.scatter_3d(
        df, x='UMAP1', y='UMAP2', z='UMAP3',
        color='Color',
        color_continuous_scale='viridis',
        opacity=0.7,
        title=title
    )
    
    # Improve layout
    fig.update_layout(
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Make points smaller for better visualization
    fig.update_traces(marker=dict(size=3))
    
    return fig