import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from nn.model import mlp_init, mlp_apply
from nn.layers import dense_init, dense_apply, dropout_apply
from nn.losses import cross_entropy_loss
from nn.optim import adam
from nn.activations import relu
from train.loop import train_step

def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist['data'].to_numpy(), mnist['target'].to_numpy().astype(int)
    
    # Normalize and split
    X = X / 255.0
    split = 60000
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, y_train, X_test, y_test

def create_batches(X, y, batch_size, rng_key):
    n = len(X)
    perm = jax.random.permutation(rng_key, n)
    X_shuffled = X[perm]
    y_shuffled = y[perm]
    
    for i in range(0, n, batch_size):
        batch_X = X_shuffled[i:i+batch_size]
        batch_y = y_shuffled[i:i+batch_size]
        batch_y_onehot = jax.nn.one_hot(batch_y, 10)
        yield jnp.array(batch_X), jnp.array(batch_y_onehot)

def evaluate(params_list, apply_fns, states_list, model_apply_wrapper, X, y, batch_size=1000):
    """Evaluate model accuracy"""
    correct = 0
    total = 0
    for i in range(0, len(X), batch_size):
        batch_X = jnp.array(X[i:i+batch_size])
        batch_y = y[i:i+batch_size]
        
        logits, _ = model_apply_wrapper(
            params_list, 
            batch_X, 
            states_list=states_list,
            training=False,
            rng=None
        )
        preds = jnp.argmax(logits, axis=-1)
        correct += jnp.sum(preds == batch_y)
        total += len(batch_y)
    
    return float(correct / total)

def visualize_predictions(params_list, apply_fns, states_list, model_apply_wrapper, X_test, y_test, num_samples=16):
    """Visualize random test samples with predictions"""
    # Randomly select samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    samples = X_test[indices]
    true_labels = y_test[indices]
    
    # Get predictions
    logits, _ = model_apply_wrapper(
        params_list,
        jnp.array(samples),
        states_list=states_list,
        training=False,
        rng=None
    )
    preds = jnp.argmax(logits, axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('MNIST Classification Results', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        # Reshape to 28x28 image
        image = samples[idx].reshape(28, 28)
        ax.imshow(image, cmap='gray')
        
        pred_label = int(preds[idx])
        true_label = int(true_labels[idx])
        confidence = float(probs[idx, pred_label])
        
        # Green if correct, red if wrong
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f'Pred: {pred_label} ({confidence:.2%})\nTrue: {true_label}', 
                     color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to 'mnist_predictions.png'")
    plt.close()

def plot_training_history(train_losses, test_accs):
    """Plot training loss and test accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss over Epochs', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, [acc * 100 for acc in test_accs], 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy over Epochs', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("✓ Saved training history to 'training_history.png'")
    plt.close()

def main():
    # Hyperparameters - adjusted for stability
    batch_size = 128
    learning_rate = 0.0005  # Lower learning rate
    num_epochs = 10
    hidden_sizes = [784, 256, 128, 10]
    dropout_rate = 0.2
    
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Initialize model
    print("\nInitializing model...")
    key = random.PRNGKey(42)
    key, init_key = random.split(key)
    
    params_list, states_list, apply_fns = mlp_init(
        key=init_key,
        sizes=hidden_sizes,
        dense_init_fn=dense_init,
        dense_init_kwargs=None
    )
    
    print(f"Model architecture: {hidden_sizes}")
    print(f"Number of layers: {len(params_list)}")
    
    # Initialize optimizer
    opt = adam(learning_rate)
    opt_state = opt.init(params_list)
    
    # Create training step function
    def model_apply_wrapper(params, x, **kwargs):
        return mlp_apply(
            params_list=params,
            x=x,
            apply_fns=apply_fns,
            activation=relu,
            dropout_rate=dropout_rate,
            dropout_apply_fn=dropout_apply,
            **kwargs
        )
    
    step_fn = train_step(
        model_apply=model_apply_wrapper,
        loss_fn=cross_entropy_loss,
        optimizer=opt,
        from_logits=True
    )
    
    # Training loop with tracking
    print("\nStarting training...")
    train_losses = []
    test_accs = []
    
    for epoch in range(num_epochs):
        key, epoch_key = random.split(key)
        epoch_loss = 0
        num_batches = 0
        
        for batch in create_batches(X_train, y_train, batch_size, epoch_key):
            key, step_key = random.split(key)
            params_list, states_list, opt_state, metrics = step_fn(
                params=params_list,
                state=states_list,
                batch=batch,
                opt_state=opt_state,
                rng=step_key
            )
            
            # Check for NaN
            if jnp.isnan(metrics['loss']):
                print(f"\n⚠ NaN detected at epoch {epoch+1}, batch {num_batches+1}")
                print(f"Stopping training early...")
                break
            
            epoch_loss += metrics['loss']
            num_batches += 1
        
        if jnp.isnan(epoch_loss):
            break
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(float(avg_loss))
        
        # Evaluate on test set
        test_acc = evaluate(params_list, apply_fns, states_list, model_apply_wrapper, X_test, y_test)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    print("\nTraining complete!")
    print(f"Final Test Accuracy: {test_accs[-1]:.4f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_predictions(params_list, apply_fns, states_list, model_apply_wrapper, X_test, y_test)
    plot_training_history(train_losses, test_accs)
    
    print("\n✓ All done!")

if __name__ == "__main__":
    main()

