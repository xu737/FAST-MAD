import torch 
import torch.nn.functional as F
import numpy as np
def get_next_factor(value, n=100):

    factors = [i for i in range(1, n+1) if n % i == 0]
    for factor in factors:
        if factor > value:
            return factor
    return n  

def adjust_periods(top_k_periods, n=100):
    adjusted_periods = [get_next_factor(period.item(), n) for period in top_k_periods]
    return torch.tensor(adjusted_periods)

def extract_top_k_periods(xb, periods, normalize=False):
    k = periods
    time_series = xb 
    if isinstance(time_series, np.ndarray):
        time_series = torch.from_numpy(time_series)
   
    if normalize:
        time_series = time_series - time_series.mean(dim=0, keepdim=True)

    fft_result = torch.fft.fft(time_series, dim=0)
    n = time_series.size(0)
    freq = torch.fft.fftfreq(n)

    amplitude = torch.abs(fft_result)
    average_amplitude = amplitude.mean(dim=1)

    positive_freq_mask = freq > 0
    # print('positive_freq_mask',positive_freq_mask)
    positive_freq = freq[positive_freq_mask]
    positive_amplitude = average_amplitude[positive_freq_mask]

    top_k_indices = torch.topk(positive_amplitude, k, dim = 0).indices
    top_k_frequencies = positive_freq[top_k_indices]

    top_k_periods = 1 / top_k_frequencies
    top_k_periods = top_k_periods[~torch.isnan(top_k_periods)] 
    adjusted_periods = adjust_periods(top_k_periods)

    return top_k_frequencies, adjusted_periods

def pearson_correlation_matrix(vector_list):
    if len(vector_list) == 1:
        return np.array([1]) 
    vectors_stacked = np.stack(vector_list)
    reshaped_vectors = vectors_stacked.reshape(vectors_stacked.shape[0], -1)
    

    pearson_matrix = np.corrcoef(reshaped_vectors, rowvar=True)
    corr_sum = np.nansum(pearson_matrix, axis=1)
    # print('corr_sum',corr_sum)
    return corr_sum

from scipy.spatial.distance import pdist, squareform

def cosine_similarity_matrix(vector_list):
    shapes = [v.shape for v in vector_list]
    if len(set(shapes)) != 1:
        raise ValueError(f"All elements in vector_list must have the same shape, but got shapes: {shapes}")

    vectors_stacked = np.stack(vector_list) 
    reshaped_vectors = vectors_stacked.reshape(vectors_stacked.shape[0], -1)
    
    if np.any(np.linalg.norm(reshaped_vectors, axis=1) == 0):
        raise ValueError("One or more rows in reshaped_vectors are zero vectors, leading to division by zero in cosine similarity.")
    
    cosine_distances = pdist(reshaped_vectors, metric='cosine')
    cosine_similarity = 1 - squareform(cosine_distances)
    
    np.fill_diagonal(cosine_similarity, np.nan)
    
    diff_max_min = np.nanmax(cosine_similarity, axis=1) - np.nanmin(cosine_similarity, axis=1)
    
    return diff_max_min


def calculate_weights(diff_max_min, top_percentage=0.4):

    if np.all(diff_max_min == diff_max_min[0]):

        return np.ones_like(diff_max_min) / len(diff_max_min)

    num_vectors = len(diff_max_min)
    top_count = max(1, int(num_vectors * top_percentage)) 


    sorted_indices = np.argsort(-diff_max_min)  
    top_indices = sorted_indices[:top_count]
    remaining_indices = sorted_indices[top_count:]

    weights = np.zeros(num_vectors)

    weights[top_indices] = 0.5 / top_count
    weights[remaining_indices] = 0.5 / (num_vectors - top_count)

    weights /= weights.sum()

    return weights


def select_representative_vectors(all_x_enc, k=3, grid_size=0.2):
 
    all_x_enc_tensor = torch.stack(all_x_enc)  
    feature_vectors = all_x_enc_tensor.view(len(all_x_enc), -1) 

    min_values = feature_vectors.min(dim=0).values
    max_values = feature_vectors.max(dim=0).values

    grid_ranges = (max_values - min_values) / grid_size
    grid_bins = (grid_ranges.ceil()).int()

    grid_indices = ((feature_vectors - min_values) / grid_size).floor().int()

    unique_grids = torch.unique(grid_indices, dim=0)
    selected_vectors = []

    if len(unique_grids) <= k:
        for grid in unique_grids:
            mask = (grid_indices == grid).all(dim=1)
            vectors_in_grid = all_x_enc_tensor[mask]
            selected_vectors.append(vectors_in_grid[0]) 
    else:

        grid_centers = (unique_grids.float() + 0.5) * grid_size + min_values
        distances = torch.norm(feature_vectors.unsqueeze(0) - grid_centers.unsqueeze(1), dim=-1)

        while len(selected_vectors) < k:

            min_distances, min_indices = torch.min(distances, dim=1)
            farthest_index = torch.argmax(min_distances)
            selected_vectors.append(all_x_enc_tensor[farthest_index])

            distances[:, farthest_index] = float('inf')


    return selected_vectors[:k]
def create_patch(xb, yb, patch_len, stride):

    xb = torch.tensor(xb)
    yb = torch.tensor(yb)

    seq_len = xb.shape[0]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1  
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[s_begin:, :]                                                    
    yb = yb[s_begin:] 
    
    xb = xb.unfold(dimension=0, size=patch_len, step=stride)                
    # print(len(yb))
    yb = yb.unfold(dimension=0, size=patch_len, step=stride) 

    # xb = xb.numpy()
    # yb = yb.numpy()

    return xb, yb, num_patch



def random_masking(xb, mask_ratio): #

    xb = xb.unsqueeze(0)  

    bs, L, nvars, D = xb.shape   
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio)) 
        
    noise = torch.rand(bs, L, nvars,device=xb.device) 
    ids_shuffle = torch.argsort(noise, dim=1)  
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     

    ids_keep = ids_shuffle[:, :len_keep, :]                                                   
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))    
   
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)                 
    # y_removed = torch.zeros(bs, L - len_keep, nvars, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) 

    mask = torch.ones([bs, L, nvars], device=x.device)                                  
    mask[:, :len_keep, :] = 0

    mask = torch.gather(mask, dim=1, index=ids_restore)                                  

    x_masked = x_masked.squeeze(0)
    x_kept = x_kept.squeeze(0)
    mask = mask.squeeze(0)
    ids_restore = ids_restore.squeeze(0)
    
    
    return x_masked, x_kept, mask, ids_restore


