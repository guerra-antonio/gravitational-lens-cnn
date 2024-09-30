import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_epsilon(epsilon) -> None:
    # Extract the first, second, and third rows of the input matrix 'epsilon'
    e1 = epsilon[0, :]  # First image (epsilon_1)
    e2 = epsilon[1, :]  # Second image (epsilon_2)
    delta_e = epsilon[2, :]  # Third image (Delta epsilon)

    # Create a figure with three subplots, 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Setting figure size to 15x5
    
    # Find the minimum and maximum values across all three matrices
    vmin = min(e1.min(), e2.min(), delta_e.min())  # Minimum value for consistent colormap scaling
    vmax = max(e1.max(), e2.max(), delta_e.max())  # Maximum value for consistent colormap scaling

    # Set the colormap to 'viridis' (a blue-to-yellow colormap)
    cmap = 'viridis'

    # Display the first image (epsilon_1) in the first subplot
    im1 = axs[0].imshow(e1, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title(r'$\epsilon_{1}$')  # Set the title for the first subplot

    # Display the second image (epsilon_2) in the second subplot
    im2 = axs[1].imshow(e2, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title(r'$\epsilon_{2}$')  # Set the title for the second subplot

    # Display the third image (Delta epsilon) in the third subplot
    im3 = axs[2].imshow(delta_e, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2].set_title(r'$\Delta\epsilon$')  # Set the title for the third subplot

    # Add a color bar to the figure, shared across all subplots
    cbar = fig.colorbar(im1, ax=axs, orientation='horizontal', fraction=0.02, pad=0.1)
    cbar.set_label(r'$\epsilon$')  # Label the color bar

    # Add a global title for the entire figure
    fig.suptitle('Input images for the problem', fontsize=16)

    # Display the plot
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Optional: Adjust layout if needed
    plt.show()  # Show the figure

def plot_kappa(kappa) -> None:
    # Create a figure with one subplot, setting the figure size to 5x5
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    # Set the colormap to 'viridis' (a gradient from blue to yellow)
    cmap = 'viridis'
    
    # Display the 'kappa' matrix as an image in the subplot
    # Set the color scaling range to the min and max values of 'kappa'
    im = ax.imshow(kappa, cmap=cmap, vmin=kappa.min(), vmax=kappa.max())
    
    # Set the title of the subplot
    ax.set_title('Expected output image for the problem')
    
    # Add a color bar to the figure, associated with the 'kappa' image
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1)
    
    # Label the color bar with the symbol for 'kappa'
    cbar.set_label(r'$\kappa$')
    
    # Display the figure
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Optional: Adjust layout if needed
    plt.show()  # Show the plot

# The above python.functions are kindly provided from the competition organizers

def metric(predicted, target):
    # Calculate a weight matrix 'w', where each element is 1 plus the absolute value of 'target'
    # divided by the maximum value in 'target'. This adds a higher weight to larger target values.
    w = 1 + torch.abs(target) / (target.max())
    
    # Compute the Mean Squared Error (MSE) between 'predicted' and 'target'
    MSE = (predicted - target) ** 2
    
    # Element-wise multiplication of the weight matrix 'w' with the MSE matrix
    ls = torch.multiply(w, MSE)
    
    # Return the sum of the weighted squared errors as the final metric
    return torch.sum(ls)