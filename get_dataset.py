import os
import torch
import numpy as np
import sys
import random
import trimesh
from itertools import combinations

import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.sparse.linalg import eigsh
import glob

os.system("clear")
import time
start_time = time.time()

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def extract_mesh_features(mesh, features=None, normalize=True):
    """
    Extract multiple features from a mesh and combine them into a single feature tensor.
    Args:
        mesh (trimesh.Trimesh): The input mesh.
        features (list): List of feature names to extract. Options include:
                        ['coordinates', 'normals', 'dihedral_angles', 'gaussian_curvature']
                        If None, all features will be extracted.
        normalize (bool): Whether to normalize the mesh vertices to [-0.5, 0.5] range.

    Returns:
        torch.Tensor: Combined feature tensor of shape [num_vertices, num_features]
    """
    # Define available features
    all_features = [
        'coordinates',
        'normals',
        'dihedral_angles',
        'gaussian_curvature'
    ]

    # Use all features if none specified
    if features is None:
        features = all_features

    # Validate feature names
    for feature in features:
        if feature not in all_features:
            raise ValueError(f"Unknown feature: {feature}. Available features: {all_features}")

    # Normalize mesh if requested
    if normalize:
        mesh = normalize_mesh(mesh)

    # Initialize empty list to store feature tensors
    feature_tensors = []

    # Extract each requested feature
    for feature in features:
        if feature == 'coordinates':
            coords = get_vertex_coordinates(mesh)
            feature_tensors.append(torch.tensor(coords, dtype=torch.float32))

        elif feature == 'normals':
            normals = get_vertex_normals(mesh)
            feature_tensors.append(torch.tensor(normals, dtype=torch.float32))

        elif feature == 'dihedral_angles':
            dihedral = generate_dihedral_angles(mesh)
            feature_tensors.append(torch.tensor(dihedral, dtype=torch.float32))

        elif feature == 'gaussian_curvature':
            gaussian = generate_gaussian_curvature(mesh)
            feature_tensors.append(gaussian)  # Already a torch tensor

    # Concatenate all feature tensors along the feature dimension
    combined_features = torch.cat(feature_tensors, dim=1)

    return combined_features

def normalize_mesh(mesh):
    """
    Normalize mesh vertices to the range [-0.5, 0.5].
    Args:
        mesh (trimesh.Trimesh): The input mesh.
    Returns:
        trimesh.Trimesh: Normalized mesh.
    """
    # Create a copy to avoid modifying the original
    mesh = mesh.copy()

    # Center the mesh
    center = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2
    mesh.vertices = mesh.vertices - center

    # Scale to [-0.5, 0.5]
    scale = mesh.vertices.max() * 2
    mesh.vertices = mesh.vertices / scale

    return mesh

def get_vertex_coordinates(mesh):
    """
    Get mesh vertex coordinates.
    Args:
        mesh (trimesh.Trimesh): The input mesh.
    Returns:
        numpy.ndarray: Vertex coordinates of shape [num_vertices, 3].
    """
    return mesh.vertices

def get_vertex_normals(mesh):
    """
    Get or compute mesh vertex normals.
    Args:
        mesh (trimesh.Trimesh): The input mesh.
    Returns:
        numpy.ndarray: Vertex normals of shape [num_vertices, 3].
    """
    # Force computation of vertex normals if not already present
    mesh.vertex_normals
    return mesh.vertex_normals

def generate_dihedral_angles(mesh):
    """
    Calculate dihedral angles for each vertex.
    Args:
        mesh (trimesh.Trimesh): The input mesh.
    Returns:
        numpy.ndarray: Dihedral angles of shape [num_vertices, 3].
    """
    # Create vertex-face adjacency matrix
    vertex_faces_adjacency_matrix = np.zeros((mesh.vertices.shape[0], mesh.faces.shape[0]))
    for vertex, faces in enumerate(mesh.vertex_faces):
        for i, face in enumerate(faces):
            if face == -1:
                break
            vertex_faces_adjacency_matrix[vertex, face] = 1

    # Calculate dihedral angles for each face
    dihedral_angle = [[] for _ in range(mesh.faces.shape[0])]
    face_adjacency = mesh.face_adjacency

    for adj_faces in face_adjacency:
        angle = np.abs(np.dot(mesh.face_normals[adj_faces[0]], mesh.face_normals[adj_faces[1]]))
        dihedral_angle[adj_faces[0]].append(angle)
        dihedral_angle[adj_faces[1]].append(angle)

    # Ensure each face has exactly 3 dihedral angles (pad if necessary)
    for i, angles in enumerate(dihedral_angle):
        while len(angles) < 3:
            angles.append(1.0)
        if len(angles) > 3:
            angles = angles[:3]
        dihedral_angle[i] = angles

    face_dihedral_angle = np.array(dihedral_angle)

    # Map face dihedral angles to vertices
    V_dihedral_angles = np.dot(vertex_faces_adjacency_matrix, face_dihedral_angle)

    return V_dihedral_angles

def generate_gaussian_curvature(mesh):
    """
    Calculate Gaussian curvature for each vertex.
    Args:
        mesh (trimesh.Trimesh): The input mesh.
    Returns:
        torch.Tensor: Normalized Gaussian curvature of shape [num_vertices, 1].
    """
    # Calculate discrete Gaussian curvature
    mesh_gaussian_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, 0)

    # Check for invalid values
    if np.isnan(mesh_gaussian_curvature).sum() > 0 or np.isinf(mesh_gaussian_curvature).sum() > 0:
        print("Warning: Gaussian curvature contains invalid values. Replacing with zeros.")
        mesh_gaussian_curvature = np.nan_to_num(mesh_gaussian_curvature, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize the Gaussian curvature
    gaussian_curvature = torch.exp(-(torch.from_numpy(np.array(mesh_gaussian_curvature)).type(torch.float)))

    # Check for invalid values after transformation
    if torch.isnan(gaussian_curvature).sum() > 0 or torch.isinf(gaussian_curvature).sum() > 0:
        print('Warning: Transformed Gaussian curvature contains invalid values. Replacing with zeros.')
        gaussian_curvature = torch.nan_to_num(gaussian_curvature, nan=0.0, posinf=0.0, neginf=0.0)

    # Min-max normalization
    if gaussian_curvature.max() > gaussian_curvature.min():
        gaussian_curvature = ((gaussian_curvature - gaussian_curvature.min()) /
                             (gaussian_curvature.max() - gaussian_curvature.min())).unsqueeze(1)
    else:
        gaussian_curvature = torch.zeros_like(gaussian_curvature).unsqueeze(1)

    return gaussian_curvature

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def compute_adjacency_matrix(num_vertices, faces):
    """
    Compute the adjacency matrix from a list of triangular faces.

    Parameters:
    - num_vertices (int): Number of vertices in the mesh.
    - faces (np.ndarray): Array of shape (num_faces, 3) containing vertex indices of each triangle.

    Returns:
    - adjacency_matrix (scipy.sparse.csr_matrix): Sparse adjacency matrix of the mesh.
    """
    row, col = [], []

    # Each triangle face [v1, v2, v3] defines edges (v1-v2, v2-v3, v3-v1)
    for v1, v2, v3 in faces:
        edges = [(v1, v2), (v2, v3), (v3, v1)]
        for i, j in edges:
            row.append(i)
            col.append(j)
            row.append(j)  # Add symmetric connection
            col.append(i)

    # Create a sparse adjacency matrix
    data = np.ones(len(row), dtype=np.float32)
    adjacency_matrix = sp.csr_matrix((data, (row, col)), shape=(num_vertices, num_vertices))

    return adjacency_matrix

def compute_mesh_gft(features_tensor, adjacency_matrix, k=None):
    """
    Apply Graph Fourier Transform on mesh features using normalized Laplacian.

    Parameters:
    -----------
    features_tensor  : numpy.ndarray
        Feature tensor with shape [Nbr_Vertices, nbr_features]
    adjacency_matrix : scipy.sparse.csr_matrix
        Sparse adjacency matrix of the mesh
    k : int, optional
        Number of eigenvectors to compute. If None, all eigenvectors are computed.
        For resource-constrained devices, using a small k is recommended.

    Returns:
    --------
    gft_coefficients : numpy.ndarray
        GFT coefficients with shape [Nbr_Vertices, nbr_features] or [k, nbr_features] if k is specified
    eigenvalues : numpy.ndarray
        Eigenvalues of the normalized Laplacian
    eigenvectors : numpy.ndarray
        Eigenvectors of the normalized Laplacian
    """
    # Extract dimensions
    num_vertices = features_tensor.shape[0]

    # Ensure adjacency matrix is in CSR format for efficient operations
    if not sp.isspmatrix_csr(adjacency_matrix):
        adjacency_matrix = sp.csr_matrix(adjacency_matrix)

    # Compute degree matrix
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()

    # Avoid division by zero for isolated vertices
    degrees[degrees == 0] = 1

    # Compute D^(-1/2)
    d_inv_sqrt = sp.diags(np.power(degrees, -0.5))

    # Compute normalized Laplacian: L_norm = I - D^(-1/2) * A * D^(-1/2)
    identity = sp.eye(num_vertices, format='csr')
    normalized_laplacian = identity - d_inv_sqrt @ adjacency_matrix @ d_inv_sqrt

    # Determine number of eigenvalues/vectors to compute
    if k is None:
        k = num_vertices
    else:
        # Ensure k is valid
        k = min(k, num_vertices - 1)

    # Compute eigenvalues and eigenvectors of normalized Laplacian
    # Using ARPACK which is efficient for large sparse matrices
    # 'SM' means we get the smallest eigenvalues first, which are most important for GFT
    eigenvalues, eigenvectors = eigsh(normalized_laplacian, k=k, which='SM')

    # Sort by eigenvalues (usually already sorted, but ensuring consistency)
    idx          = eigenvalues.argsort()
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Apply the Graph Fourier Transform: GFT(x) = U^T * x
    # where U is the matrix of eigenvectors
    gft_coefficients = np.matmul(eigenvectors.T, features_tensor)

    return gft_coefficients #, eigenvalues, eigenvectors

def apply_spectral_filtering(features_tensor, adjacency_matrix, filter_function, k=None):
    """
    Apply spectral filtering on mesh features using GFT.

    Parameters:
    -----------
    features_tensor  : numpy.ndarray
        Feature tensor with shape [Nbr_Vertices, nbr_features]
    adjacency_matrix : scipy.sparse.csr_matrix
        Sparse adjacency matrix of the mesh
    filter_function  : callable
        Function that takes eigenvalues and returns filter coefficients
    k : int, optional
        Number of eigenvectors to compute

    Returns:
    --------
    filtered_features : numpy.ndarray
        Filtered features tensor
    """
    # Apply GFT
    gft_coeffs, eigenvalues, eigenvectors = compute_mesh_gft(features_tensor, adjacency_matrix, k)

    # Apply filter in spectral domain
    filter_coeffs = filter_function(eigenvalues)
    filtered_gft  = gft_coeffs * filter_coeffs[:, np.newaxis]

    return filtered_gft

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def load_mesh_with_features(mesh_path, features=None, k=25, normalize=True, use_normalized_laplacian=True):
    """
    Load a mesh from file, extract features, and compute spectral features.

    Args:
        mesh_path (str) : Path to the mesh file.
        features (list) : List of feature names to extract.
        normalize (bool): Whether to normalize the mesh.
        num_eigenvectors (int)         : Number of eigenvectors to use for GFT.
        use_normalized_laplacian (bool): Whether to use normalized Laplacian.

    Returns:
        tuple: (mesh, features_tensor, spectral_features, eigenvalues, eigenvectors)
    """

    # Load the mesh
    mesh = trimesh.load(mesh_path)

    # Extract standard features
    features_tensor = extract_mesh_features(mesh, features)

    # Compute spectral features
    adjacency_matrix  = compute_adjacency_matrix(mesh.vertices.shape[0], mesh.faces)
    spectral_features = compute_mesh_gft(features_tensor, adjacency_matrix, k=k)

    return mesh, spectral_features

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Dataset Loader 
def load_dataset_single_subject(data_path, subject_index=0, num_positive_pairs=300, num_negative_pairs=60, k=25, train_ratio=0.7, val_ratio=0.15):
    """
    Load the dataset and generate labeled pairs for training, focusing on a single subject.
    Args:
        data_path (str)         : Path to BU-3DFE dataset directory.
        subject_index (int)     : Index of the subject to use as the primary subject.
        num_positive_pairs (int): Number of positive pairs to generate for the subject.
        num_negative_pairs (int): Number of negative pairs to generate for the subject.
        train_ratio (float)     : Ratio of data to use for training.
        val_ratio (float)       : Ratio of data to use for validation.
    Returns:
        tuple: (train_data, val_data, test_data) where each is a list of
               (x1, edge_index1, x2, edge_index2, label) tuples.
    """
    # Get all subject directories

    # For BU_3DFE dataset
    #subject_dirs = sorted([d for d in os.listdir(data_path) if d.startswith('F') or d.startswith('M')])

    # FaceScape
    subject_dirs = sorted(glob.glob(data_path+'/*'))

    num_subjects = len(subject_dirs)
    print(f"Number of subjects: {num_subjects}")

    # Validate the subject index
    if subject_index >= num_subjects:
        subject_index = subject_index % num_subjects
        print(f"Subject index out of range. Using index {subject_index} instead.")

    # Select the target subject
    subject_dir  = subject_dirs[subject_index]
    subject_path = os.path.join(data_path, subject_dir)
    mesh_files   = sorted([f for f in os.listdir(subject_path) if f.endswith('.obj')])
    print(f"Processing subject {subject_dir} with {len(mesh_files)} meshes.")

    # Check if we have enough meshes
    if len(mesh_files) < 2:
        raise ValueError(f"Subject {subject_dir} has only {len(mesh_files)} meshes. At least 2 are required for positive pairs.")

    all_data     = []
    lst_features = ['coordinates', 'normals', 'dihedral_angles', 'gaussian_curvature']

    # Load all meshes for this subject
    subject_meshes = []
    for mesh_file in mesh_files:
        try:
            mesh_path = os.path.join(subject_path, mesh_file)

            #*******************************************************************
            #mesh              = trimesh.load(mesh_path)
            #vertices_features = torch.tensor(mesh.vertices, dtype=torch.float32) # use (x,y,z)
            mesh, vertices_features = load_mesh_with_features(mesh_path, features=lst_features, k=k)
            #*******************************************************************

            edges     = torch.tensor(mesh.faces[:, [0, 1, 1, 2, 2, 0]].T, dtype=torch.long).t().contiguous()
            subject_meshes.append((vertices_features, edges, mesh_path))
        except Exception as e:
            print(f"Error loading {mesh_file}: {e}")
            continue

    # Calculate how many positive pairs we can create
    max_possible_pairs    = len(subject_meshes) * (len(subject_meshes) - 1) // 2
    actual_positive_pairs = min(max_possible_pairs, num_positive_pairs)
    if actual_positive_pairs < num_positive_pairs:
        print(f"Warning: Requested {num_positive_pairs} positive pairs, but only {actual_positive_pairs} are possible with {len(subject_meshes)} meshes.")

    # Generate positive pairs (same subject, different expressions)
    possible_positive_pairs = list(combinations(range(len(subject_meshes)), 2))
    if len(possible_positive_pairs) > actual_positive_pairs:
        positive_pair_indices = random.sample(possible_positive_pairs, actual_positive_pairs)
    else:
        positive_pair_indices = possible_positive_pairs

    for idx1, idx2 in positive_pair_indices:
        x1, edge_index1, path1 = subject_meshes[idx1]
        x2, edge_index2, path2 = subject_meshes[idx2]
        all_data.append((x1, edge_index1, x2, edge_index2, 1))  # Label 1 for positive pair
        #print(f"Created positive pair: {os.path.basename(path1)} - {os.path.basename(path2)}")

    #---------------------------------------------------------------------------
    # Generate negative pairs (different subjects)
    #print("-" * 50)
    other_subjects = list(range(num_subjects))
    other_subjects.remove(subject_index)  # Remove current subject

    # Ensure we don't request more negative pairs than available other subjects
    actual_negative_pairs = min(len(other_subjects), num_negative_pairs)
    if actual_negative_pairs < num_negative_pairs:
        print(f"Warning: Requested {num_negative_pairs} negative pairs, but only {actual_negative_pairs} other subjects are available.")
        negative_subject_indices = other_subjects + random.choices(other_subjects, k=num_negative_pairs - len(other_subjects))
    else:
        negative_subject_indices = random.sample(other_subjects, num_negative_pairs)


    random.shuffle(negative_subject_indices)
    total      = len(negative_subject_indices)
    size_train = int(total * train_ratio)
    size_val   = int(total * (train_ratio + val_ratio))
    
    train_idx_neg = negative_subject_indices[:size_train]
    val_idx_neg   = negative_subject_indices[size_train:size_val]
    test_idx_neg  = negative_subject_indices[size_val:]

    train_data_neg = []
    val_data_neg   = []
    test_data_neg  = []

    lst_neg_subj_indx = []
    lst_neg_subj_indx.append(train_idx_neg)
    lst_neg_subj_indx.append(val_idx_neg)
    lst_neg_subj_indx.append(test_idx_neg)
    
    act_set = 0
    for negative_subject_indices_local in lst_neg_subj_indx:
        # For each selected mesh from our subject, create a negative pair with another subject
        for neg_subject_idx in negative_subject_indices_local:
            neg_subject_dir  = subject_dirs[neg_subject_idx]
            neg_subject_path = os.path.join(data_path, neg_subject_dir)
            neg_mesh_files   = [f for f in os.listdir(neg_subject_path) if f.endswith('.obj')]

            if not neg_mesh_files:
                print(f"No mesh files found for subject {neg_subject_dir}, skipping.")
                continue


            # Select a random mesh from this subject
            #chosen_mesh_file = random.choice(neg_mesh_files)

            lst_neg_subj = random.sample(neg_mesh_files, 5) # x 5 for each subj

            for chosen_mesh_file in lst_neg_subj:
                try:
                    neg_mesh_path  = os.path.join(neg_subject_path, chosen_mesh_file)

                    #*******************************************************************
                    #neg_mesh       = trimesh.load(neg_mesh_path)
                    #x_neg          = torch.tensor(neg_mesh.vertices, dtype=torch.float32)
                    neg_mesh, x_neg = load_mesh_with_features(neg_mesh_path, features=lst_features, k=k)
                    #*******************************************************************

                    edge_index_neg = torch.tensor(neg_mesh.faces[:, [0, 1, 1, 2, 2, 0]].T, dtype=torch.long).t().contiguous()

                    # Use a random mesh from our subject to pair with
                    idx = random.randrange(len(subject_meshes))
                    x_subj, edge_index_subj, subj_path = subject_meshes[idx]

                    if act_set == 0:                     
                        train_data_neg.append((x_subj, edge_index_subj, x_neg, edge_index_neg, 0))  # Label 0 for negative pair
                        #print("Train")
                        #print(f"Created negative pair: {os.path.basename(subj_path)} - {neg_subject_dir}/{chosen_mesh_file}")
                    elif act_set == 1:
                        val_data_neg.append((x_subj, edge_index_subj, x_neg, edge_index_neg, 0))  # Label 0 for negative pair
                        #print("Val")
                        #print(f"Created negative pair: {os.path.basename(subj_path)} - {neg_subject_dir}/{chosen_mesh_file}")
                    elif act_set == 2:
                        test_data_neg.append((x_subj, edge_index_subj, x_neg, edge_index_neg, 0))  # Label 0 for negative pair
                        #print("Test")
                        #print(f"Created negative pair: {os.path.basename(subj_path)} - {neg_subject_dir}/{chosen_mesh_file}")
                        
                except Exception as e:
                    print(f"Error processing negative pair with {neg_subject_dir}: {e}")
                    continue
                   
        act_set = act_set + 1

    # Shuffle all data
    random.shuffle(all_data)

    # Split into train, validation, and test sets
    n = len(all_data)
    train_idx = int(n * train_ratio)
    val_idx   = int(n * (train_ratio + val_ratio))

    train_data = all_data[:train_idx]
    val_data   = all_data[train_idx:val_idx]
    test_data  = all_data[val_idx:]

    train_data = train_data + train_data_neg
    val_data   = val_data + val_data_neg
    test_data  = test_data + test_data_neg
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)


    
    print("-" * 50)
    print(f"Generated dataset with {len(all_data)} total pairs:")
    print(f"  * Training  : {len(train_data)} pairs")
    print(f"  * Validation: {len(val_data)} pairs")
    print(f"  * Testing   : {len(test_data)} pairs")

    """
    print("One Example : ")
    print("train_data : ",train_data[0][0].shape, type(train_data[0][0]))
    print("train_data : ",train_data[0][1].shape, type(train_data[0][1]))
    print("train_data : ",train_data[0][2].shape, type(train_data[0][2]))
    print("train_data : ",train_data[0][3].shape, type(train_data[0][3]))
    print("train_data : ",train_data[0][4], type(train_data[0][4]))
    """
    return train_data, val_data, test_data

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Dataset path
data_path    = "...your path..." # BU_3DFE or FaceScape

dataset_name = "BU_3DFE" # BU_3DFE FaceScape

nbr_subjects = 100
k            = 10

for i in range(nbr_subjects):
    print("-" * 50)
    print("Client : ",i)
    train_data, val_data, test_data = load_dataset_single_subject(data_path,
                                    subject_index=i,
                                    num_positive_pairs=600, 
                                    num_negative_pairs=120, # x 5 for each subj 
                                    k=k,
                                    train_ratio=0.7, val_ratio=0.15)
    
    # Save to an .pth file
    torch.save(train_data, "../preprocessed_dataset/"+str(dataset_name)+"_k"+str(k)+"_c"+str(nbr_subjects)+"/train_data_"+str(i)+".pth")
    torch.save(val_data,   "../preprocessed_dataset/"+str(dataset_name)+"_k"+str(k)+"_c"+str(nbr_subjects)+"/val_data_"+str(i)+".pth")
    torch.save(test_data,  "../preprocessed_dataset/"+str(dataset_name)+"_k"+str(k)+"_c"+str(nbr_subjects)+"/test_data_"+str(i)+".pth")
    


print("-" * 50)
end_time = time.time()
print("The execution time (in seconds) is : ",end_time-start_time)
print("Done ...")

