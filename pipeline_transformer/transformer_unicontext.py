import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import f1_score

class PositionalEncoding(nn.Module):
    """Encodage positionnel pour les séquences d'entrée du transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Créer la matrice d'encodage positionnel
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Stocker au format batch-first (1, max_len, d_model) pour l'usage du module
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Étendre/déplacer l'encodage positionnel si nécessaire (séquences plus longues ou changement de device)
        if seq_len > self.pe.size(1) or self.pe.device != x.device:
            device = x.device
            new_len = seq_len
            pe = torch.zeros(new_len, self.d_model, device=device)
            position = torch.arange(0, new_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        return x + self.pe[:, :seq_len, :]

class MultiHeadAttention(nn.Module):
    """Mécanisme d'attention multi-têtes."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Projections linéaires et reshape pour l'attention multi-têtes
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Appliquer l'attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concaténer les têtes
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Transformation linéaire finale
        output = self.w_o(attention_output)
        return output

class FeedForward(nn.Module):
    """Réseau feed-forward appliqué position par position."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Bloc transformer avec auto-attention et couches feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Auto-attention avec connexion résiduelle et normalisation de couche
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward avec connexion résiduelle et normalisation de couche
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class UsernameTransformer(nn.Module):
    """
    Modèle Transformer pour prédire l'identifiant utilisateur à partir des séquences
    d'actions avec contexte de navigateur.

    - En entraînement: prend des triplets (username, action_sequence, browser).
    - En inférence: prend (action_sequence, browser) et prédit l'utilisateur.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1000,
        dropout: float = 0.1,
        n_usernames: int = None,
        n_browsers: int = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_usernames = n_usernames
        self.n_browsers = n_browsers
        
        # Embedding des tokens pour les séquences d'actions
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Embedding du navigateur (contexte)
        self.browser_embedding = nn.Embedding(n_browsers, d_model)
        
        # Embedding de l'utilisateur (pour l'entraînement)
        self.username_embedding = nn.Embedding(n_usernames, d_model)
        
        # Encodage positionnel (augmenté de 1 pour inclure le token navigateur)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len + 1)
        
        # Blocs Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Couches de sortie
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Tête de classification pour la prédiction d'utilisateur (MLP)
        if n_usernames is not None:
            self.username_classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_usernames)
            )
        
        # Initialiser les poids
        self.init_weights()
    
    def init_weights(self):
        """Initialiser les poids du modèle."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq):
        """Créer un masque de padding pour les séquences de longueur variable."""
        return (seq != 0).unsqueeze(1).unsqueeze(2)
    
    def forward(self, action_sequence, browser=None, username=None, training=True):
        """
        Passe avant du transformer avec contexte navigateur.

        Arguments:
            action_sequence: tenseur (batch_size, seq_len) d'IDs de tokens d'actions.
            browser: tenseur (batch_size,) d'IDs de navigateurs.
            username: tenseur (batch_size,) d'IDs d'utilisateurs (utilisé uniquement en entraînement).
            training: booléen indiquant le mode entraînement.

        Retourne:
            En entraînement: (logits_utilisateur, embeddings_actions)
            En inférence: logits_utilisateur
        """
        batch_size, seq_len = action_sequence.shape
        
        # Créer l'embedding du token navigateur
        browser_emb = self.browser_embedding(browser) * math.sqrt(self.d_model)  # (batch_size, d_model)
        browser_emb = browser_emb.unsqueeze(1)  # (batch_size, 1, d_model)     

        # Embeddings des tokens d'actions
        action_emb = self.token_embedding(action_sequence) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        
        # Concaténer le token navigateur au début de la séquence
        x = torch.cat([browser_emb, action_emb], dim=1)  # (batch_size, seq_len + 1, d_model)
        
        # Créer le masque de padding (étendu pour inclure le token navigateur)
        action_mask = self.create_padding_mask(action_sequence)  # (batch_size, 1, 1, seq_len)
        browser_mask = torch.ones(batch_size, 1, 1, 1, device=action_sequence.device)  # Browser token is never masked
        mask = torch.cat([browser_mask, action_mask], dim=-1)  # (batch_size, 1, 1, seq_len + 1)
        
        # Ajouter l'encodage positionnel
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Passage dans les blocs Transformer
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Appliquer la normalisation de couche
        x = self.layer_norm(x)
        
        # Moyenne globale (pooling) sur la séquence en excluant le token navigateur
        # Utiliser le masque pour ignorer les tokens de padding et exclure le navigateur du pooling
        mask_expanded = mask.squeeze(1).squeeze(1).float()  # (batch_size, seq_len + 1)
        # Ne moyenner que sur les tokens d'action (ignorer le navigateur en position 0)
        action_tokens = x[:, 1:, :]  # (batch_size, seq_len, d_model)
        action_mask = mask_expanded[:, 1:]  # (batch_size, seq_len)
        x_masked = action_tokens * action_mask.unsqueeze(-1)
        denom = action_mask.sum(dim=1, keepdim=True)
        zero_mask = (denom == 0)
        denom = denom.clamp(min=1.0)
        pooled = x_masked.sum(dim=1) / denom
        pooled = pooled.masked_fill(zero_mask, 0.0)
        
        if training and username is not None:
            # Mode entraînement: retourner les logits utilisateur et les embeddings
            username_logits = self.username_classifier(pooled)
            return username_logits, pooled
        else:
            # Mode inférence: retourner uniquement les logits utilisateur
            username_logits = self.username_classifier(pooled)
            return username_logits
    
    def predict_username(self, action_sequence, browser=None):
        """
        Prédire l'utilisateur à partir de la séquence d'actions et du navigateur (mode inférence).

        Arguments:
            action_sequence: tenseur (batch_size, seq_len) ou (seq_len,)
            browser: tenseur (batch_size,) ou scalaire d'IDs de navigateurs

        Retourne:
            Logits et probabilités prédits pour l'utilisateur
        """
        self.eval()
        
        if action_sequence.dim() == 1:
            action_sequence = action_sequence.unsqueeze(0)
        
        if browser is not None and browser.dim() == 0:
            browser = browser.unsqueeze(0)
        
        with torch.no_grad():
            username_logits = self.forward(action_sequence, browser, training=False)
            username_probs = F.softmax(username_logits, dim=-1)
            
        return username_logits, username_probs

class UsernameTransformerTrainer:
    """Utilitaires d'entraînement pour le UsernameTransformer."""
    
    def __init__(self, model, learning_rate=1e-5, device='cpu', class_weights=None):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Utiliser des poids de classe si fournis, sinon CrossEntropyLoss standard
        if class_weights is not None:
            class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, action_sequences: torch.Tensor, usernames: torch.Tensor, browsers: torch.Tensor = None):
        """Une étape d'entraînement."""
        self.model.train()
        
        action_sequences = action_sequences.to(self.device)
        usernames = usernames.to(self.device)
        browsers = browsers.to(self.device)
        
        self.optimizer.zero_grad()
        
        username_logits, _ = self.model(action_sequences, browsers, usernames, training=True)
        loss = self.criterion(username_logits, usernames)
        
        loss.backward()
        # Vérifier les gradients avant l'étape d'optimisation
        # check_gradient_flow(self.model)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, action_sequences, usernames, browsers=None):
        """Évaluer le modèle sur les données de validation."""
        self.model.eval()
        
        action_sequences = action_sequences.to(self.device)
        usernames = usernames.to(self.device)
        browsers = browsers.to(self.device)
        
        with torch.no_grad():
            username_logits = self.model(action_sequences, browsers, training=False)
            loss = self.criterion(username_logits, usernames)
            
            predictions = torch.argmax(username_logits, dim=-1)
            accuracy = (predictions == usernames).float().mean()
            
            # Calculer le score F1 macro
            predictions_cpu = predictions.cpu().numpy()
            usernames_cpu = usernames.cpu().numpy()
            macro_f1 = f1_score(usernames_cpu, predictions_cpu, average='macro', zero_division=0)
        
        return loss.item(), accuracy.item(), macro_f1, predictions

# Fonction utilitaire pour calculer les poids de classe
def calculate_class_weights(username_tokens):
    """
    Calculer des poids de classe pour un jeu déséquilibré, façon « balanced » de sklearn.

    Arguments:
        username_tokens: liste des IDs de tokens d'utilisateurs.

    Retourne:
        class_weights: tenseur des poids pour chaque classe.
    """
    username_counts = torch.bincount(torch.tensor(username_tokens, dtype=torch.long))
    total_samples = len(username_tokens)
    num_classes = len(username_counts)
    
    # Poids équilibrés sklearn: n_samples / (n_classes * np.bincount(y))
    # Pas de normalisation ici: PyTorch gère l'échelle; des poids forts aident en fort déséquilibre
    class_weights = total_samples / (num_classes * username_counts.float())
    
    return class_weights

# Example usage and training function
def create_model(vocab_size, n_usernames, n_browsers=None, **kwargs):
    """Créer un modèle UsernameTransformer avec des hyperparamètres par défaut."""
    return UsernameTransformer(
        vocab_size=vocab_size,
        n_usernames=n_usernames,
        n_browsers=n_browsers,
        d_model=kwargs.get('d_model', 64),
        n_heads=kwargs.get('n_heads', 2),
        n_layers=kwargs.get('n_layers', 2),
        d_ff=kwargs.get('d_ff', 512),
        max_seq_len=kwargs.get('max_seq_len', 500),
        dropout=kwargs.get('dropout', 0.15)
    )

def train_model(model, train_data, val_data, learning_rate=1e-5, epochs=50, batch_size=8, max_seq_len=100, device='cpu'):
    """
    Entraîner le modèle UsernameTransformer avec contexte navigateur.

    Arguments:
        model: instance de UsernameTransformer.
        train_data: tuple (action_sequences, usernames, browsers) ou (action_sequences, usernames).
        val_data: tuple (action_sequences, usernames, browsers) ou (action_sequences, usernames).
        epochs: nombre d'époques d'entraînement.
        batch_size: taille de lot (réduite pour éviter des problèmes mémoire).
        max_seq_len: longueur maximale de séquence à traiter.
        device: périphérique pour l'entraînement.
    """
    username_tokens = train_data[1]
    weights = calculate_class_weights(username_tokens)
    trainer = UsernameTransformerTrainer(model, learning_rate=learning_rate, device=device, class_weights=weights)
    
    train_sequences, train_usernames, train_browsers = train_data
    val_sequences, val_usernames, val_browsers = val_data
    
    print(f"Training on {len(train_sequences)} samples")
    print(f"Validation on {len(val_sequences)} samples")
    print(f"Using max sequence length: {max_seq_len}")
    print(f"Using batch size: {batch_size}")
    
    # Fonction pour tronquer les séquences à max_seq_len
    def truncate_sequences(sequences, max_len):
        truncated = []
        for seq in sequences:
            if len(seq) > max_len:
                # Conserver les max_len derniers tokens (actions les plus récentes)
                truncated.append(seq[-max_len:])
            else:
                truncated.append(seq)
        return truncated
    
    # Tronquer les séquences pour éviter des problèmes de mémoire
    train_sequences = truncate_sequences(train_sequences, max_seq_len)
    val_sequences = truncate_sequences(val_sequences, max_seq_len)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_sequences), batch_size):
            batch_sequences = train_sequences[i:i+batch_size]
            batch_usernames = train_usernames[i:i+batch_size]
            batch_browsers = train_browsers[i:i+batch_size]
            
            # Convert to tensors and pad sequences
            batch_sequences = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(seq, dtype=torch.long) for seq in batch_sequences],
                batch_first=True,
                padding_value=0
            )
            
            batch_usernames = torch.tensor(batch_usernames, dtype=torch.long)
            
            batch_browsers = torch.tensor(batch_browsers, dtype=torch.long)
            
            loss = trainer.train_step(batch_sequences, batch_usernames, batch_browsers)
            train_loss += loss
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # Validation - par lots pour éviter les problèmes mémoire
        val_loss = 0
        val_accuracy = 0
        num_val_batches = 0
        all_predictions = []
        all_true_labels = []
        
        for i in range(0, len(val_sequences), batch_size):
            val_batch_sequences = val_sequences[i:i+batch_size]
            val_batch_usernames = val_usernames[i:i+batch_size]
            val_batch_browsers = val_browsers[i:i+batch_size]
            
            # Conversion en tenseurs et padding des séquences
            val_batch_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(seq, dtype=torch.long) for seq in val_batch_sequences],
                batch_first=True,
                padding_value=0
            )
            
            val_batch_usernames = torch.tensor(val_batch_usernames, dtype=torch.long)
            
            val_batch_browsers = torch.tensor(val_batch_browsers, dtype=torch.long)
            
            batch_loss, batch_accuracy, batch_macro_f1, batch_predictions = trainer.evaluate(val_batch_tensor, val_batch_usernames, val_batch_browsers)
            val_loss += batch_loss
            val_accuracy += batch_accuracy
            all_predictions.append(batch_predictions.cpu())
            all_true_labels.append(val_batch_usernames.cpu())
            num_val_batches += 1
        
        val_loss = val_loss / num_val_batches
        val_accuracy = val_accuracy / num_val_batches
        
        # Calculer le F1 macro global sur tout le jeu de validation
        all_predictions = torch.cat(all_predictions, dim=0)
        all_true_labels = torch.cat(all_true_labels, dim=0)
        val_macro_f1 = f1_score(all_true_labels.numpy(), all_predictions.numpy(), average='macro', zero_division=0)
        # Déterminer la longueur minimale pour bincount
        minlength = model.n_usernames if model.n_usernames is not None else (all_predictions.max().item() + 1 if len(all_predictions) > 0 else 1)
        prediction_counts = torch.bincount(all_predictions, minlength=minlength)
        total_predictions = len(all_predictions)
        prediction_proportions = prediction_counts.float() / total_predictions
        
        # Obtenir les utilisateurs les plus prédits (uniquement ceux réellement prédits)
        non_zero_mask = prediction_counts > 0
        if non_zero_mask.any():
            top_k = min(5, non_zero_mask.sum().item())
            top_proportions, top_indices = torch.topk(prediction_proportions, k=top_k)
        else:
            top_proportions = torch.tensor([])
            top_indices = torch.tensor([], dtype=torch.long)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Macro-F1: {val_macro_f1:.4f}")
            if len(top_indices) > 0:
                print(f"  Utilisateurs les plus prédits (par fréquence):")
                for idx, (username_idx, proportion) in enumerate(zip(top_indices.tolist(), top_proportions.tolist())):
                    print(f"    {idx+1}. Username {username_idx}: {proportion:.2%} ({prediction_counts[username_idx].item()}/{total_predictions})")
    
    return model




def check_gradient_flow(model):
    """Vérifier que les gradients se propagent correctement dans le modèle."""
    print("\n=== Gradient Flow Check ===")
    total_norm = 0
    param_norm = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Vérifier si un gradient existe
            if param.grad is None:
                print(f"⚠️  {name}: NO GRADIENT (None)")
                zero_grad_count += 1
                continue
            
            # Vérifier la norme du gradient
            param_norm = param.data.norm().item()
            grad_norm = param.grad.norm().item()
            
            if grad_norm == 0:
                print(f"⚠️  {name}: ZERO gradient (norm=0)")
                zero_grad_count += 1
            elif grad_norm < 1e-7:
                print(f"⚠️  {name}: VANISHING gradient (norm={grad_norm:.2e})")
            elif grad_norm > 100:
                print(f"⚠️  {name}: EXPLODING gradient (norm={grad_norm:.2e})")
            else:
                print(f"✓  {name}: OK (grad_norm={grad_norm:.4f}, param_norm={param_norm:.4f})")
            
            # Ratio gradient/paramètre
            ratio = grad_norm / (param_norm + 1e-10)
            if ratio < 1e-6:
                print(f"   ⚠️  Very small gradient/param ratio: {ratio:.2e}")
            
            total_norm += grad_norm ** 2
    
    total_norm = total_norm ** 0.5
    print(f"\nTotal gradient norm: {total_norm:.4f}")
    print(f"Parameters with zero/None gradients: {zero_grad_count}")
    
    return total_norm, zero_grad_count