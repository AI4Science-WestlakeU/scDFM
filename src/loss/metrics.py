import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


# TODO: To be implemented into flow generation
class FlowMatchingMetrics:

    def __init__(self, device='cuda'):
        self.device = device
        
    def reconstruction_metrics(self, generated, target):

        if isinstance(generated, torch.Tensor):
            generated = generated.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
            
        mae = np.mean(np.abs(generated - target))
        mse = np.mean((generated - target) ** 2)
        rmse = np.sqrt(mse)
        
        
        pos_mask = target > 0
        zero_mask = target == 0
        
        pos_mae = np.mean(np.abs(generated[pos_mask] - target[pos_mask])) if pos_mask.any() else 0.0
        zero_mae = np.mean(np.abs(generated[zero_mask] - target[zero_mask])) if zero_mask.any() else 0.0
        
        
        r2 = r2_score(target.flatten(), generated.flatten())
        
        rel_error = np.mean(np.abs((generated - target) / (target + 1e-8)))
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'pos_mae': pos_mae,
            'zero_mae': zero_mae,
            'r2_score': r2,
            'relative_error': rel_error
        }
    
    def distribution_metrics(self, generated, target):
        if isinstance(generated, torch.Tensor):
            generated = generated.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
            
        metrics = {}

        mean_diff = np.mean(np.abs(np.mean(generated, axis=0) - np.mean(target, axis=0)))
        var_diff = np.mean(np.abs(np.var(generated, axis=0) - np.var(target, axis=0)))
        
        metrics['mean_difference'] = mean_diff
        metrics['variance_difference'] = var_diff
        
        
        correlations = []
        for i in range(generated.shape[1]):
            if np.var(generated[:, i]) > 1e-8 and np.var(target[:, i]) > 1e-8:
                corr, _ = stats.pearsonr(generated[:, i], target[:, i])
                if not np.isnan(corr):
                    correlations.append(corr)
        
        metrics['mean_correlation'] = np.mean(correlations) if correlations else 0.0
        
    
        kl_divs = []
        for i in range(min(generated.shape[1], 100)): 
            try:
                bins = np.linspace(
                    min(generated[:, i].min(), target[:, i].min()),
                    max(generated[:, i].max(), target[:, i].max()),
                    50
                )
                hist_gen, _ = np.histogram(generated[:, i], bins=bins, density=True)
                hist_target, _ = np.histogram(target[:, i], bins=bins, density=True)
                
        
                hist_gen = hist_gen + 1e-8
                hist_target = hist_target + 1e-8
                
            
                hist_gen = hist_gen / np.sum(hist_gen)
                hist_target = hist_target / np.sum(hist_target)
                
                kl_div = stats.entropy(hist_gen, hist_target)
                if not np.isnan(kl_div) and not np.isinf(kl_div):
                    kl_divs.append(kl_div)
            except:
                continue
                
        metrics['mean_kl_divergence'] = np.mean(kl_divs) if kl_divs else float('inf')
        
        js_divs = []
        for i in range(min(generated.shape[1], 100)):
            try:
                bins = np.linspace(
                    min(generated[:, i].min(), target[:, i].min()),
                    max(generated[:, i].max(), target[:, i].max()),
                    50
                )
                hist_gen, _ = np.histogram(generated[:, i], bins=bins, density=True)
                hist_target, _ = np.histogram(target[:, i], bins=bins, density=True)
                
                hist_gen = hist_gen + 1e-8
                hist_target = hist_target + 1e-8
                hist_gen = hist_gen / np.sum(hist_gen)
                hist_target = hist_target / np.sum(hist_target)
                
                js_div = jensenshannon(hist_gen, hist_target)
                if not np.isnan(js_div):
                    js_divs.append(js_div)
            except:
                continue
                
        metrics['mean_js_divergence'] = np.mean(js_divs) if js_divs else 1.0
        
        return metrics
    
    def embedding_metrics(self, generated, target):

        if isinstance(generated, torch.Tensor):
            generated = generated.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
            
        metrics = {}
        
    
        cosine_sims = []
        for i in range(generated.shape[0]):
            sim = np.dot(generated[i], target[i]) / (
                np.linalg.norm(generated[i]) * np.linalg.norm(target[i]) + 1e-8
            )
            cosine_sims.append(sim)
        
        metrics['mean_cosine_similarity'] = np.mean(cosine_sims)
        
 
        euclidean_dists = np.linalg.norm(generated - target, axis=1)
        metrics['mean_euclidean_distance'] = np.mean(euclidean_dists)
        

        if generated.shape[0] > 10: 
            k = min(5, generated.shape[0] // 2)
            
            nn_target = NearestNeighbors(n_neighbors=k)
            nn_target.fit(target)
            _, target_indices = nn_target.kneighbors(target)
            
            nn_generated = NearestNeighbors(n_neighbors=k)
            nn_generated.fit(generated)
            _, generated_indices = nn_generated.kneighbors(generated)
            
            preservation = 0
            for i in range(generated.shape[0]):
                intersection = len(set(target_indices[i]) & set(generated_indices[i]))
                preservation += intersection / k
            
            metrics['neighbor_preservation'] = preservation / generated.shape[0]
        else:
            metrics['neighbor_preservation'] = 0.0
            
        return metrics
    
    def flow_specific_metrics(self, generated, target, flow_model=None):

        metrics = {}
        
        if isinstance(generated, torch.Tensor):
            generated = generated.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
            
       
        pairwise_dists = []
        for i in range(min(generated.shape[0], 100)):
            for j in range(i + 1, min(generated.shape[0], 100)):
                dist = np.linalg.norm(generated[i] - generated[j])
                pairwise_dists.append(dist)
        
        metrics['generation_diversity'] = np.mean(pairwise_dists) if pairwise_dists else 0.0
        
        if generated.shape[0] > 10 and target.shape[0] > 10:
            
            combined = np.vstack([generated, target])
            pca = PCA(n_components=min(10, combined.shape[1]))
            combined_pca = pca.fit_transform(combined)
            
            generated_pca = combined_pca[:generated.shape[0]]
            target_pca = combined_pca[generated.shape[0]:]
            
            
            coverage = 0
            for dim in range(generated_pca.shape[1]):
                gen_min, gen_max = generated_pca[:, dim].min(), generated_pca[:, dim].max()
                target_min, target_max = target_pca[:, dim].min(), target_pca[:, dim].max()
                
                if target_max > target_min:
                    overlap = max(0, min(gen_max, target_max) - max(gen_min, target_min))
                    coverage += overlap / (target_max - target_min)
            
            metrics['mode_coverage'] = coverage / generated_pca.shape[1]
        else:
            metrics['mode_coverage'] = 0.0
            
        return metrics
    
    def comprehensive_evaluation(self, generated, target, embeddings_generated=None, embeddings_target=None):

        all_metrics = {}
        
        recon_metrics = self.reconstruction_metrics(generated, target)
        all_metrics.update({f'recon_{k}': v for k, v in recon_metrics.items()})
        
        
        dist_metrics = self.distribution_metrics(generated, target)
        all_metrics.update({f'dist_{k}': v for k, v in dist_metrics.items()})
        
        flow_metrics = self.flow_specific_metrics(generated, target)
        all_metrics.update({f'flow_{k}': v for k, v in flow_metrics.items()})
        
        if embeddings_generated is not None and embeddings_target is not None:
            emb_metrics = self.embedding_metrics(embeddings_generated, embeddings_target)
            all_metrics.update({f'emb_{k}': v for k, v in emb_metrics.items()})
        
        return all_metrics

def evaluate_flow_generation(generated_data, target_data, generated_embeddings=None, target_embeddings=None, device='cuda'):

    metrics = FlowMatchingMetrics(device=device)
    return metrics.comprehensive_evaluation(
        generated_data, target_data, 
        generated_embeddings, target_embeddings
    )

def print_metrics(metrics_dict, title="Metrics"):

    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    
    categories = {
        'Reconstruction': [k for k in metrics_dict.keys() if k.startswith('recon_')],
        'Distribution': [k for k in metrics_dict.keys() if k.startswith('dist_')],
        'Flow-specific': [k for k in metrics_dict.keys() if k.startswith('flow_')],
        'Embedding': [k for k in metrics_dict.keys() if k.startswith('emb_')]
    }
    
    for category, keys in categories.items():
        if keys:
            print(f"\n{category}:")
            print("-" * 20)
            for key in keys:
                clean_key = key.replace(f"{category.lower().replace('-', '_')}_", "")
                value = metrics_dict[key]
                if isinstance(value, float):
                    print(f"  {clean_key:<20}: {value:.6f}")
                else:
                    print(f"  {clean_key:<20}: {value}")
    
    print(f"{'='*50}\n")


def test_metrics():

    batch_size, n_features = 100, 2000
    generated = torch.randn(batch_size, n_features)
    target = torch.randn(batch_size, n_features)
    
    metrics = evaluate_flow_generation(generated, target)
    
    print_metrics(metrics, "Flow Matching Evaluation")
    
    return metrics


def pairwise_sq_dists(X, Y):
    # X:[m,d], Y:[n,d] -> [m,n]
    return torch.cdist(X, Y, p=2)**2

@torch.no_grad()
def median_sigmas(X, scales=(0.5, 1.0, 2.0, 4.0)):
    Z = X
    D2 = pairwise_sq_dists(Z, Z)
    tri = D2[~torch.eye(D2.size(0), dtype=bool, device=D2.device)]
    m = torch.median(tri).clamp_min(1e-12)          
    s2 = torch.tensor(scales, device=Z.device) * m   
    sigmas = torch.sqrt(s2)                         
    return [float(s.item()) for s in sigmas]

def mmd2_unbiased_multi_sigma(X, Y, sigmas):
    """
    """
    m, n = X.size(0), Y.size(0)
    Dxx = pairwise_sq_dists(X, X)   # [m,m]
    Dyy = pairwise_sq_dists(Y, Y)   # [n,n]
    Dxy = pairwise_sq_dists(X, Y)   # [m,n]

    vals = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * (sigma ** 2) + 1e-12)
        Kxx = torch.exp(-beta * Dxx)
        Kyy = torch.exp(-beta * Dyy)
        Kxy = torch.exp(-beta * Dxy)

        
        term_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1) + 1e-12)
        term_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1) + 1e-12)
        term_xy = Kxy.mean()  # / (m*n)
        vals.append(term_xx + term_yy - 2.0 * term_xy)

    return torch.stack(vals).mean()

if __name__ == "__main__":
    test_metrics()
