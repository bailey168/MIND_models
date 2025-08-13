def extract_upper_triangle(matrix):
    """Extracts the upper triangle of a symmetric matrix."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]

def get_top_edges(upper_triangle, percentage=0.5):
    """Returns the top edges based on the specified percentage of the highest magnitudes."""
    threshold_index = int(len(upper_triangle) * percentage)
    top_edges = np.argsort(upper_triangle)[-threshold_index:]
    return top_edges

def one_hot_encode(labels, num_classes):
    """Performs one-hot encoding for the given labels."""
    return np.eye(num_classes)[labels]