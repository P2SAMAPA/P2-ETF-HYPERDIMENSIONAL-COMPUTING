import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_hypervector(dim, seed=None):
    """Generate a random bipolar hypervector (-1, +1)."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice([-1, 1], size=dim)

def quantize_returns(returns, n_levels=5):
    """Quantize continuous returns into n_levels discrete levels."""
    if len(returns) == 0:
        return np.array([])
    # Use quantiles for equal-frequency binning
    quantiles = np.linspace(0, 100, n_levels + 1)[1:-1]
    bins = np.percentile(returns, quantiles)
    # If bins are degenerate, fallback to equal-width
    if len(np.unique(bins)) == 1:
        bins = np.linspace(returns.min(), returns.max(), n_levels + 1)[1:-1]
    return np.digitize(returns, bins)

def level_hypervector(level, n_levels, dim):
    """Generate a level hypervector for a given quantization level."""
    # Each level gets a random hypervector (or use a systematic construction)
    # For simplicity, we use a permutation-based approach:
    # Each level is a random hypervector
    seed = level * 12345
    return generate_hypervector(dim, seed=seed)

def bundle(vectors):
    """Bundle (majority sum) multiple hypervectors."""
    if len(vectors) == 0:
        return None
    return np.sign(np.sum(vectors, axis=0))

def bind(u, v):
    """Bind (element-wise multiply) two hypervectors."""
    return u * v

def permutation(v, shift=1):
    """Permutation (circular shift) of a hypervector."""
    return np.roll(v, shift)

def similarity(u, v):
    """Cosine similarity between two hypervectors."""
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
        return 0.0
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def hd_score(returns, macro_df, dim=10000, n_levels=5):
    """
    Compute HD computing score for a single ETF.
    The score is the similarity between the current market state and the "up" prototype.
    """
    if len(returns) < 5 or macro_df is None or len(macro_df) < 5:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 5:
        return 0.0
    # Quantize returns
    quantized = quantize_returns(returns, n_levels)
    # Generate level hypervectors for each quantization level
    level_hvs = [level_hypervector(i, n_levels, dim) for i in range(n_levels)]
    # Build item memories for each time step
    # For each time step, we bundle the level hypervector with macro-derived vectors
    # For simplicity, we use a composite vector: bind(level_hv, macro_factor_hv)
    # Compute macro factor (first principal component of macro variables)
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    macro_factor = pca.fit_transform(macro_scaled).flatten()
    macro_factor = (macro_factor - macro_factor.min()) / (macro_factor.max() - macro_factor.min() + 1e-8)
    # Quantize macro factor
    macro_quantized = quantize_returns(macro_factor, n_levels)
    # Generate macro level hypervectors
    macro_level_hvs = [level_hypervector(i + 1000, n_levels, dim) for i in range(n_levels)]
    # Build a prototype for "up" days (returns > 0)
    up_vectors = []
    down_vectors = []
    for i, ret in enumerate(returns):
        level_idx = quantized[i]
        macro_level_idx = macro_quantized[i]
        # Composite vector = bind(level_hv, macro_level_hv)
        composite = bind(level_hvs[level_idx], macro_level_hvs[macro_level_idx])
        if ret > 0:
            up_vectors.append(composite)
        else:
            down_vectors.append(composite)
    # Prototype vectors
    up_prototype = bundle(up_vectors) if up_vectors else generate_hypervector(dim, seed=999)
    down_prototype = bundle(down_vectors) if down_vectors else generate_hypervector(dim, seed=888)
    # Current state (last time step)
    last_level = quantized[-1]
    last_macro_level = macro_quantized[-1]
    current_state = bind(level_hvs[last_level], macro_level_hvs[last_macro_level])
    # Score = similarity to up_prototype - similarity to down_prototype
    sim_up = similarity(current_state, up_prototype)
    sim_down = similarity(current_state, down_prototype)
    score = sim_up - sim_down
    return float(score)
