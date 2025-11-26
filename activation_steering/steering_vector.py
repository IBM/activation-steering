import dataclasses
import os, json
import typing
import warnings

import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FastICA
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import matplotlib.pyplot as plt

from activation_steering.malleable_model import MalleableModel, get_model_layer_list
from activation_steering.utils import ContrastivePair, custom_progress
from activation_steering.steering_dataset import SteeringDataset
from activation_steering.config import log, GlobalConfig



@dataclasses.dataclass
class SteeringVector:
    """
    A dataclass representing a steering vector used for guiding the language model.

    Attributes:
        model_type: The type of the model this vector is associated with.
        directions: A dictionary mapping layer IDs to numpy arrays of directions.
        explained_variances: A dictionary of explained variances.
    """
    model_type: str
    directions: dict[int, np.ndarray]
    explained_variances: dict

    @classmethod
    def train(cls, model: MalleableModel | PreTrainedModel, tokenizer: PreTrainedTokenizerBase, steering_dataset: SteeringDataset, **kwargs) -> "SteeringVector":
        """
        Train a SteeringVector for a given model and tokenizer using the provided dataset.

        Args:
            model: The model to train the steering vector for.
            tokenizer: The tokenizer associated with the model.
            steering_dataset: The dataset to use for training.
            **kwargs: Additional keyword arguments.

        Returns:
            A new SteeringVector instance.
        """
        log("Training steering vector", class_name="SteeringVector")
        # Set the pad_token_id of the tokenizer to 0
        tokenizer.pad_token_id = 0
        
        dirs, variances = read_representations(
            model,
            tokenizer,
            steering_dataset.formatted_dataset,
            suffixes=steering_dataset.suffixes,
            **kwargs,
        )
        
        return cls(model_type=model.config.model_type, 
                   directions=dirs, 
                   explained_variances=variances)
    
    def save(self, file_path: str):
        """
        Save the SteeringVector to a file.

        Args:
            file_path: The path to save the file to. If it doesn't end with '.svec', 
                       this extension will be added.
        """
        if not file_path.endswith('.svec'):
            file_path += '.svec'
        
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            log(f"Created directory: {directory}", class_name="SteeringVector")
        

        log(f"Saving SteeringVector to {file_path}", class_name="SteeringVector")
        data = {
            "model_type": self.model_type,
            "directions": {k: v.tolist() for k, v in self.directions.items()},
            "explained_variances": self.explained_variances
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)
        log(f"SteeringVector saved successfully", class_name="SteeringVector")

    @classmethod
    def load(cls, file_path: str) -> "SteeringVector":
        """
        Load a SteeringVector from a file.

        Args:
            file_path: The path to load the file from. If it doesn't end with '.svec', 
                       this extension will be added.

        Returns:
            A new SteeringVector instance loaded from the file.
        """
        if not file_path.endswith('.svec'):
            file_path += '.svec'
        
        log(f"Loading SteeringVector from {file_path}", class_name="SteeringVector")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        directions = {int(k): np.array(v) for k, v in data["directions"].items()}
        explained_variances = {int(k): v for k, v in data["explained_variances"].items()}
        
        log(f"Loaded directions for layers: {list(directions.keys())}", class_name="SteeringVector")
        log(f"Shape of first direction vector: {next(iter(directions.values())).shape}", class_name="SteeringVector")
        
        return cls(model_type=data["model_type"], 
               directions=directions, 
               explained_variances=explained_variances)


def read_representations(model: MalleableModel | PreTrainedModel, tokenizer: PreTrainedTokenizerBase, inputs: list[ContrastivePair], hidden_layer_ids: typing.Iterable[int] | None = None, batch_size: int = 32, method: typing.Literal["pca_diff", "pca_center", "pca_pairwise","mean_diff","linear_probe","lda","mass_mean_shift","pls","ica","svd_primary","sparse_pca","random","kernel_pca","kmeans_diff"] = "pca_pairwise", save_analysis: bool = False, output_dir: str = "activation_steering_figures", accumulate_last_x_tokens: typing.Union[int, str] = 1, suffixes: typing.List[typing.Tuple[str, str]] = None) -> dict[int, np.ndarray]:
    """
    Extract representations from the language model based on the contrast dataset.

    Args:
        model: The model to extract representations from.
        tokenizer: The tokenizer associated with the model.
        inputs: A list of ContrastivePair inputs.
        hidden_layer_ids: The IDs of hidden layers to extract representations from.
        batch_size: The batch size to use when processing inputs.
        method: The method to use for preparing training data ("pca_diff", "pca_center", or "pca_pairwise").
        save_analysis: Whether to save PCA analysis figures.
        output_dir: The directory to save analysis figures to.
        accumulate_last_x_tokens: How many tokens to accumulate for the hidden state.
        suffixes: List of suffixes to use when accumulating hidden states.

    Returns:
        A dictionary mapping layer IDs to numpy arrays of directions.
    """
    log(f"Reading representations for {len(inputs)} inputs", class_name="SteeringVector")
    # If hidden_layer_ids is not provided, use all layers of the model
    if hidden_layer_ids is None:
        hidden_layer_ids = range(model.config.num_hidden_layers)

    if accumulate_last_x_tokens == "all":
        # Accumulate hidden states of all tokens
        log("... accumulating all hidden states", class_name="SteeringVector")
    elif accumulate_last_x_tokens == "suffix-only":
        log(f"... accumulating suffix-only hidden states", class_name="SteeringVector")
    else:
        log(f"... accumulating last {accumulate_last_x_tokens} hidden states", class_name="SteeringVector")
        
    # Get the total number of layers in the model
    n_layers = len(get_model_layer_list(model))
    
    # Normalize the layer indexes if they are negative
    hidden_layer_ids = [i if i >= 0 else n_layers + i for i in hidden_layer_ids]
    
    # Prepare the input strings for the model by extracting the positive and negative examples from the contrastive pairs
    train_strs = [s for ex in inputs for s in (ex.positive, ex.negative)]
    
    # Example:
    # inputs = [
    #     ContrastivePair(positive="I'm feeling happy today.", negative="I'm feeling sad today."),
    #     ContrastivePair(positive="The weather is great!", negative="The weather is terrible.")
    # ]
    # train_strs = [
    #     "I'm feeling happy today.", "I'm feeling sad today.",
    #     "The weather is great!", "The weather is terrible."
    # ]
    
    # Call the batched_get_hiddens function to get the hidden states for each specified layer
    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layer_ids, batch_size, accumulate_last_x_tokens, suffixes
    )
    if save_analysis:
        save_pca_figures(layer_hiddens, hidden_layer_ids, method, output_dir, inputs)

    # Initialize an empty dictionary to store the directions for each layer
    directions: dict[int, np.ndarray] = {}
    explained_variances: dict[int, float] = {}
    
    # Iterate over each specified layer
    for layer in custom_progress(hidden_layer_ids, description="Reading Hidden Representations ..."):
        # Retrieve the hidden states for the current layer
        h = layer_hiddens[layer]
        
    # --- VECTOR EXTRACTION METHODS ---

        if method == "mean_diff":
            # Method: Mean Difference
            # Description: Calculates the simple difference between the centroids of positive 
            # and negative classes. It does not use covariance or variance information.
            # Use case: Few-shot scenarios or as a robust baseline.
            
            pos_mean = np.mean(h[::2], axis=0)
            neg_mean = np.mean(h[1::2], axis=0)
            
            direction = pos_mean - neg_mean
            
            # Normalize to unit length
            direction = direction / np.linalg.norm(direction)

            directions[layer] = direction
            # No explained variance metric for mean difference, setting to 1.0 placeholder
            explained_variances[layer] = 1.0

        elif method == "linear_probe":
            # Method: Linear Probe (Logistic Regression)
            # Description: Trains a linear classifier to separate positive and negative examples.
            # The vector is the normal to the decision boundary (coefficients).
            # Use case: Maximizing discriminative power between two distinct concepts.

            # Prepare labels: 1 for Positive, 0 for Negative
            y = np.zeros(h.shape[0])
            y[::2] = 1 
            y[1::2] = 0 

            # Train Logistic Regression
            # 'liblinear' is good for small datasets, high dimensionality
            clf = LogisticRegression(fit_intercept=True, solver='liblinear', max_iter=1000).fit(h, y)
            
            direction = clf.coef_.squeeze()
            direction = direction / np.linalg.norm(direction)
            
            directions[layer] = direction
            # Use classification accuracy as a proxy for "quality" of the vector
            explained_variances[layer] = clf.score(h, y)

        elif method == "lda":
            # Method: Linear Discriminant Analysis (Fisher's LDA)
            # Description: Finds a direction that maximizes the distance between class means 
            # while minimizing within-class variance.
            # Use case: Statistically more robust than Mean Diff when sufficient data is available.

            y = np.zeros(h.shape[0])
            y[::2] = 1
            y[1::2] = 0

            # Solver 'svd' is preferred for high-dimensional data (feature > sample)
            lda = LinearDiscriminantAnalysis(solver='svd')
            lda.fit(h, y)
            
            direction = lda.coef_.squeeze()
            direction = direction / np.linalg.norm(direction)
            
            directions[layer] = direction
            explained_variances[layer] = lda.score(h, y)

        elif method == "mass_mean_shift":
            # Method: Mass Mean Shift (Covariance-Adjusted Mean Difference)
            # Description: Adjusts the mean difference vector based on the data's covariance 
            # structure. Formula: Sigma^(-1) * (Mu_pos - Mu_neg).
            # Use case: When the concept direction is skewed by the geometry of the latent space.

            pos_mean = np.mean(h[::2], axis=0)
            neg_mean = np.mean(h[1::2], axis=0)
            delta_mu = pos_mean - neg_mean

            # Calculate Covariance Matrix (centered)
            centered_data = h - h.mean(axis=0)
            cov = np.cov(centered_data, rowvar=False)

            # Solve for direction: Cov * d = delta_mu
            try:
                direction = np.linalg.solve(cov, delta_mu)
            except np.linalg.LinAlgError:
                # Use Pseudo-Inverse if matrix is singular (common in high-dim, low-sample)
                inv_cov = np.linalg.pinv(cov)
                direction = inv_cov @ delta_mu

            direction = direction / np.linalg.norm(direction)

            directions[layer] = direction
            explained_variances[layer] = 1.0

        elif method == "svd_primary":
            # Method: SVD on Raw Data
            # Description: Performs Singular Value Decomposition on uncentered data to find 
            # the primary direction of magnitude/orientation.
            # Use case: When the origin/bias information is important to preserve.
            
            u, s, vh = np.linalg.svd(h, full_matrices=False)
            
            # First singular vector
            direction = vh[0]
            
            directions[layer] = direction
            # Explained variance ratio
            explained_variances[layer] = (s[0] ** 2) / np.sum(s ** 2)

        elif method == "ica":
            # Method: Independent Component Analysis
            # Description: Seeks statistically independent components rather than just 
            # uncorrelated ones (like PCA).
            # Use case: Disentangling mixed (polysemantic) signals/concepts.
            
            ica = FastICA(n_components=1, whiten='unit-variance', random_state=42)
            
            # ICA usually works best on difference vectors
            train_diff = h[::2] - h[1::2]
            ica.fit(train_diff)
            
            direction = ica.components_[0]
            direction = direction / np.linalg.norm(direction)
            
            directions[layer] = direction
            explained_variances[layer] = 1.0

        elif method == "pls":
            # Method: Partial Least Squares Regression
            # Description: Finds a direction that maximizes covariance between features (X) 
            # and labels (Y). Hybrid of PCA and Linear Regression.
            # Use case: Very stable for high-dimensional, low-sample settings (High Dimension Low Sample Size).

            y = np.zeros(h.shape[0])
            y[::2] = 1
            y[1::2] = 0

            pls = PLSRegression(n_components=1)
            pls.fit(h, y)
            
            # x_weights_ represent the direction in X space
            direction = pls.x_weights_.squeeze()
            direction = direction / np.linalg.norm(direction)
            
            directions[layer] = direction
            explained_variances[layer] = pls.score(h, y)

        elif method in ["pca_diff", "pca_center", "pca_pairwise"]:
            # Method: Principal Component Analysis (Variations)
            # Description: Finds the direction of maximum variance in the data.
            
            if method == "pca_diff":
                # PCA on difference vectors (Pos - Neg)
                train = h[::2] - h[1::2]
            elif method == "pca_center":
                # PCA on globally centered data
                center = h.mean(axis=0)
                train = h - center
            elif method == "pca_pairwise":
                # PCA on pairwise centered data (removes shared content per pair)
                center = (h[::2] + h[1::2]) / 2
                train = h.copy()
                train[::2] -= center
                train[1::2] -= center
            
            pca_model = PCA(n_components=1, whiten=False).fit(train)
            directions[layer] = pca_model.components_.astype(np.float32).squeeze(axis=0)
            explained_variances[layer] = pca_model.explained_variance_ratio_[0]
        elif method == "sparse_pca":
            # Method: Sparse PCA
            # Description: Similar to PCA but enforces sparsity (L1 regularization). 
            # It tries to construct the direction using as few neurons as possible.
            # Use case: "Surgical" steering. Reduces side effects by affecting fewer neurons.
            # Ideally suits the "Sparse Autoencoder" hypothesis of LLMs.

            # alpha: Regularization strength (higher = more sparse)
            # ridge_alpha: Stability parameter
            spca = SparsePCA(n_components=1, alpha=1, ridge_alpha=0.01, random_state=42)
            
            # Use difference vectors or centered data
            if h.shape[0] > 1:
                train = h[::2] - h[1::2]
            else:
                train = h

            spca.fit(train)
            
            direction = spca.components_[0]
            
            # If the vector is all zeros (too much regularization), fallback to random or small noise
            if np.all(direction == 0):
                warnings.warn(f"Layer {layer}: Sparse PCA resulted in a zero vector. Reducing sparsity might help.")
                direction = np.random.randn(direction.shape[0])

            direction = direction / np.linalg.norm(direction)
            
            directions[layer] = direction
            # Sparse PCA doesn't provide a standard explained variance ratio
            explained_variances[layer] = 1.0

        elif method == "random":
            # Method: Random Direction (Baseline)
            # Description: Generates a completely random unit vector.
            # Use case: Scientific control. Used to verify that your other methods 
            # are actually doing something better than chance.

            # Generate random vector from standard normal distribution
            direction = np.random.randn(h.shape[1])
            
            direction = direction / np.linalg.norm(direction)
            
            directions[layer] = direction
            explained_variances[layer] = 0.0
        elif method == "kernel_pca":
            # Method: Kernel PCA (RBF Kernel)
            # Description: Performs PCA in a high-dimensional feature space using the "kernel trick".
            # It can capture non-linear relationships that standard PCA misses.
            # Use case: When the concept manifold is curved (non-linear geometry).
            
            # fit_inverse_transform=True is CRITICAL. We need to map the vector back 
            # to the original activation space (pre-image problem).
            kpca = KernelPCA(n_components=1, kernel="rbf", fit_inverse_transform=True, alpha=1.0)
            
            # Use pairwise differences
            if h.shape[0] > 1:
                train = h[::2] - h[1::2]
            else:
                train = h

            kpca.fit(train)
            
            # Get the first eigenvector in the feature space
            # Note: In Kernel PCA, components_ are in the transformed space.
            # We reconstruct the "linear approximation" of that non-linear direction in our original space.
            # This is an approximation but often a very powerful one.
            direction = kpca.eigenvectors_[0] @ kpca.dual_coef_ @ kpca.X_fit_
            
            direction = direction / np.linalg.norm(direction)
            
            directions[layer] = direction
            explained_variances[layer] = 1.0 

        elif method == "kmeans_diff":
            # Method: K-Means Centroid Difference
            # Description: Instead of taking the simple mean (which is sensitive to outliers),
            # we cluster the positive and negative examples and take the difference of their
            # main cluster centers.
            # Use case: When data is "multimodal" (e.g., "Happy" has two types: Calm & Excited).
            # This method finds the "densest" representation of the concept.

            pos_data = h[::2]
            neg_data = h[1::2]

            # Find the main cluster center for positives (k=1 is robust mean)
            kmeans_pos = KMeans(n_clusters=1, n_init=10).fit(pos_data)
            pos_center = kmeans_pos.cluster_centers_[0]

            # Find the main cluster center for negatives
            kmeans_neg = KMeans(n_clusters=1, n_init=10).fit(neg_data)
            neg_center = kmeans_neg.cluster_centers_[0]
            
            direction = pos_center - neg_center
            
            direction = direction / np.linalg.norm(direction)
            
            directions[layer] = direction
            explained_variances[layer] = 1.0
        else:
            raise ValueError("Unknown method: " + method)
        # Example:
        # Suppose the hidden states for the current layer are:
        # h = np.array([
        #     [0.1, 0.2, 0.3],  # Positive example 1
        #     [0.4, 0.5, 0.6],  # Negative example 1
        #     [0.2, 0.3, 0.4],  # Positive example 2
        #     [0.5, 0.6, 0.7]   # Negative example 2
        #     [0.2, 0.3, 0.4],  # Positive example 3
        #     [0.5, 0.6, 0.7]   # Negative example 3
        # ])
        #
        # Performing PCA on the training data (differences between positive and negative examples)
        # will extract the direction vector that captures the most significant variation.
        # The resulting direction vector might be something like [0.57735027, 0.57735027, 0.57735027].
        
        # Project the hidden states onto the direction vector
        projected_hiddens = project_onto_direction(h, directions[layer])
        
        # Calculate the mean of positive examples being smaller than negative examples
        positive_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        
        # Calculate the mean of positive examples being larger than negative examples
        positive_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        
        # If positive examples are smaller on average, flip the direction vector
        if positive_smaller_mean > positive_larger_mean:
            directions[layer] *= -1
    
    # Return the dictionary mapping layer IDs to their corresponding direction vectors
    return directions, explained_variances


def batched_get_hiddens(model, tokenizer, inputs: list[str], hidden_layer_ids: list[int],batch_size: int, accumulate_last_x_tokens: typing.Union[int, str] = 1, suffixes: typing.List[typing.Tuple[str, str]] = None) -> dict[int, np.ndarray]:
    """
    Retrieve the hidden states from the specified layers of the language model for the given input strings.

    Args:
        model: The model to get hidden states from.
        tokenizer: The tokenizer associated with the model.
        inputs: A list of input strings.
        hidden_layer_ids: The IDs of hidden layers to get states from.
        batch_size: The batch size to use when processing inputs.
        accumulate_last_x_tokens: How many tokens to accumulate for the hidden state.
        suffixes: List of suffixes to use when accumulating hidden states.

    Returns:
        A dictionary mapping layer IDs to numpy arrays of hidden states.
    """
    # Split the input strings into batches based on the specified batch size
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    
    # Initialize an empty dictionary to store the hidden states for each specified layer
    hidden_states = {layer: [] for layer in hidden_layer_ids}
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Iterate over each batch of input strings
        for batch in custom_progress(batched_inputs, description="Collecting Hidden Representations ..."):
            # Pass the batch through the language model and retrieve the hidden states
            out = model(
                **tokenizer(batch, padding=True, return_tensors="pt").to(model.device),
                output_hidden_states=True,
            )
            
            # Iterate over each specified layer ID
            for layer_id in hidden_layer_ids:
                # Adjust the layer index if it is negative
                hidden_idx = layer_id + 1 if layer_id >= 0 else layer_id
                
                # Iterate over each batch of hidden states
                for i, batch_hidden in enumerate(out.hidden_states[hidden_idx]):
                    if accumulate_last_x_tokens == "all":
                        accumulated_hidden_state = torch.mean(batch_hidden, dim=0)
                    elif accumulate_last_x_tokens == "suffix-only":
                        if suffixes:
                            # Tokenize the suffix
                            suffix_tokens = tokenizer.encode(suffixes[0][0], add_special_tokens=False)
                            # Get the hidden states for the suffix tokens
                            suffix_hidden = batch_hidden[-len(suffix_tokens):, :]
                            accumulated_hidden_state = torch.mean(suffix_hidden, dim=0)
                        else:
                            warnings.warn("'suffix-only' option used but no suffixes provided. Using last token instead.")
                            accumulated_hidden_state = batch_hidden[-1, :]
                    else:
                        accumulated_hidden_state = torch.mean(batch_hidden[-accumulate_last_x_tokens:, :], dim=0)
                    
                    hidden_states[layer_id].append(accumulated_hidden_state.squeeze().cpu().numpy())
            
            
            # Delete the model output to free up memory
            del out
    
    # Stack the hidden states for each layer into a numpy array
    # Return the dictionary mapping layer IDs to their corresponding stacked hidden states
    return {k: np.vstack(v) for k, v in hidden_states.items()}


def project_onto_direction(H, direction):
    """
    Project a matrix H onto a direction vector.

    Args:
        H: The matrix to project.
        direction: The direction vector to project onto.

    Returns:
        The projected matrix.
    """
    # Calculate the magnitude (Euclidean norm) of the direction vector
    mag = np.linalg.norm(direction)
    
    # Assert that the magnitude is not infinite to ensure validity
    assert not np.isinf(mag)
    
    # Perform the projection by multiplying the matrix H with the direction vector
    # Divide the result by the magnitude of the direction vector to normalize the projection
    return (H @ direction) / mag


def save_pca_figures(layer_hiddens, hidden_layer_ids, method, output_dir, inputs):
    """
    Save PCA analysis figures for each hidden layer and create a macroscopic x-axis layer analysis plot.

    Args:
        layer_hiddens: A dictionary of hidden states for each layer.
        hidden_layer_ids: The IDs of hidden layers.
        method: The method used for preparing training data.
        output_dir: The directory to save the figures to.
        inputs: The input data used for the analysis.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lists to store the variances and layer IDs for the macroscopic analysis
    variances = []
    layers = []

    for layer in custom_progress(hidden_layer_ids, description="Saving PCA Figures"):
        h = layer_hiddens[layer]

        if method == "pca_diff":
            train = h[::2] - h[1::2]
        elif method in ["mean_diff", "linear_probe", "lda", "mass_mean_shift","pls","ica","svd_primary","sparse_pca", "random","kernel_pca", "kmeans_diff"]:
            center = (h[::2] + h[1::2]) / 2
            train = h.copy()
            train[::2] -= center
            train[1::2] -= center
        elif method == "pca_center":
            center = h.mean(axis=0)
            train = h - center
        elif method == "pca_pairwise":
            center = (h[::2] + h[1::2]) / 2
            train = h.copy()
            train[::2] -= center
            train[1::2] -= center
        else:
            raise ValueError("unknown method " + method)

        pca_model = PCA(n_components=2, whiten=False).fit(train)

        # Project the dataset points onto the first two principal components
        projected_data = pca_model.transform(h)

        # Separate the projected data into positive and negative examples
        positive_data = projected_data[::2]
        negative_data = projected_data[1::2]

        # Plot the projected points with separate colors for positive and negative examples
        plt.figure(figsize=(8, 6))
        plt.scatter(positive_data[:, 0], positive_data[:, 1], alpha=0.7, label="Positive Examples")
        plt.scatter(negative_data[:, 0], negative_data[:, 1], alpha=0.7, label="Negative Examples")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA Visualization - Layer {layer}")
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(output_dir, f"pca_layer_{layer}.png"))
        plt.close()

        # Store the variance explained by PC1 and the corresponding layer ID for the macroscopic analysis
        variances.append(pca_model.explained_variance_ratio_[0])
        layers.append(layer)

    # Create the macroscopic x-axis layer analysis plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, variances, marker='o')
    plt.xlabel("Layer ID")
    plt.ylabel("Variance Explained by PC1")
    plt.title("Macroscopic X-Axis Layer Analysis")
    plt.grid(True)
    plt.xticks(layers)
    plt.tight_layout()

    # Save the macroscopic analysis figure
    plt.savefig(os.path.join(output_dir, "macroscopic_analysis.png"))
    plt.close()

