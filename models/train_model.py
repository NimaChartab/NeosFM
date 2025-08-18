"""
NEOS Foundation Model Training Script
====================================

Train the foundation model with:
- Dual-channel ResNet (NC1+NC2 as 2 channels)
- 50 tabular features (normalized)
- Data from multiple visits
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.foundation_model import NEOSFoundationModel, ContrastiveLoss
from data.dataset import NEOSDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FoundationModelTrainer:
    """Trainer for dual-channel ResNet foundation model with 50 features"""
    
    def __init__(self, 
                 model, 
                 dataset, 
                 criterion,
                 device='cuda',
                 batch_size=128,
                 learning_rate=1e-4,
                 weight_decay=1e-4):
        
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        
        # Create data loader
        self.train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # Optimizer settings
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate * 3,  # Peak LR
            epochs=30,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.05,  # 5% warmup
            anneal_strategy='cos'
        )
        
        # Training tracking
        self.train_losses = []
        self.epoch = 0
        
        # Mixed precision for faster training
        self.scaler = torch.cuda.amp.GradScaler()
        
    def train_epoch(self):
        """Train for one epoch with mixed precision"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1} Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move data to device
                nc1_images = batch['nc1_image'].to(self.device, non_blocking=True)
                nc2_images = batch['nc2_image'].to(self.device, non_blocking=True)
                tabular_data = batch['tabular_data'].to(self.device, non_blocking=True)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    # Model processes NC1+NC2 as dual channels
                    image_embeds, tabular_embeds, fused_embeds = self.model(
                        nc1_images, nc2_images, tabular_data, return_individual=True
                    )
                    
                    # Contrastive loss between image and tabular modalities
                    # Since image_embeds already combines NC1+NC2, we use it for both image arguments
                    loss = self.criterion(image_embeds, image_embeds, tabular_embeds, strategy="image_tabular")
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{avg_loss:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                    'GPU': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
                })
                
                # Memory cleanup
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_epoch_loss = total_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        
        return avg_epoch_loss
    
    def train(self, num_epochs):
        """Train the foundation model"""
        logger.info(f"Starting foundation model training for {num_epochs} epochs...")
        logger.info(f"Architecture: Dual-Channel ResNet + 50 Tabular Features")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.train_loader.batch_size}")
        logger.info(f"Dataset size: {len(self.train_loader.dataset):,}")
        logger.info(f"Batches per epoch: {len(self.train_loader):,}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        best_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Train epoch
            epoch_loss = self.train_epoch()
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Log results
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, "
                       f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s, "
                       f"Total: {total_time/60:.1f}min")
            
            # Save best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': epoch_loss,
                    'dataset_size': len(self.train_loader.dataset),
                    'architecture': 'dual_channel_resnet_50_features'
                }, 'best_model.pt')
                logger.info(f"Saved best model with loss: {epoch_loss:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': epoch_loss,
                    'train_losses': self.train_losses,
                    'dataset_size': len(self.train_loader.dataset),
                    'architecture': 'dual_channel_resnet_50_features'
                }, f'checkpoint_epoch_{epoch+1}.pt')
            
            # GPU memory management
            torch.cuda.empty_cache()
        
        total_training_time = time.time() - start_time
        logger.info(f"Foundation model training completed!")
        logger.info(f"Total training time: {total_training_time/60:.1f} minutes")
        logger.info(f"Final loss: {self.train_losses[-1]:.4f}")
        logger.info(f"Best loss: {best_loss:.4f}")
        
        return self.train_losses
    
    def plot_training_analysis(self):
        """Plot comprehensive training analysis"""
        plt.figure(figsize=(16, 12))
        
        # Training loss
        plt.subplot(2, 3, 1)
        plt.plot(self.train_losses, 'b-', linewidth=2, label='Improved Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Improved Model Training Loss')
        plt.legend()
        plt.grid(True)
        
        # Log scale loss
        plt.subplot(2, 3, 2)
        plt.semilogy(self.train_losses, 'b-', linewidth=2, label='Log Scale Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss (Log Scale)')
        plt.legend()
        plt.grid(True)
        
        # Learning rate schedule
        plt.subplot(2, 3, 3)
        epochs = len(self.train_losses)
        lr_curve = []
        base_lr = 1e-4
        max_lr = base_lr * 3
        for epoch in range(epochs):
            if epoch < epochs * 0.05:
                lr = base_lr + (max_lr - base_lr) * (epoch / (epochs * 0.05))
            else:
                lr = base_lr + (max_lr - base_lr) * 0.5 * (1 + np.cos(np.pi * (epoch - epochs * 0.05) / (epochs * 0.95)))
            lr_curve.append(lr)
        
        plt.plot(lr_curve, 'r-', linewidth=2, label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        
        # Loss improvement per epoch
        plt.subplot(2, 3, 4)
        if len(self.train_losses) > 1:
            improvements = [0] + [self.train_losses[i-1] - self.train_losses[i] for i in range(1, len(self.train_losses))]
            plt.plot(improvements, 'g-', linewidth=2, label='Loss Improvement')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('Epoch')
            plt.ylabel('Loss Improvement')
            plt.title('Per-Epoch Loss Improvement')
            plt.legend()
            plt.grid(True)
        
        # Model architecture info
        plt.subplot(2, 3, 5)
        plt.text(0.1, 0.8, 'Improved Architecture:', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.7, '• Dual-Channel ResNet', fontsize=12)
        plt.text(0.1, 0.6, '• NC1+NC2 → Single CNN', fontsize=12)
        plt.text(0.1, 0.5, '• 50 Tabular Features', fontsize=12)
        plt.text(0.1, 0.4, '• Z-score Normalized', fontsize=12)
        plt.text(0.1, 0.3, f'• {sum(p.numel() for p in self.model.parameters()):,} params', fontsize=12)
        plt.text(0.1, 0.2, f'• {len(self.train_loader.dataset):,} samples', fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Model Details')
        
        # Final performance summary
        plt.subplot(2, 3, 6)
        final_loss = self.train_losses[-1]
        best_loss = min(self.train_losses)
        improvement = (self.train_losses[0] - final_loss) / self.train_losses[0] * 100
        
        plt.text(0.1, 0.8, 'Training Results:', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.7, f'Initial Loss: {self.train_losses[0]:.4f}', fontsize=12)
        plt.text(0.1, 0.6, f'Final Loss: {final_loss:.4f}', fontsize=12)
        plt.text(0.1, 0.5, f'Best Loss: {best_loss:.4f}', fontsize=12)
        plt.text(0.1, 0.4, f'Improvement: {improvement:.1f}%', fontsize=12)
        plt.text(0.1, 0.3, f'Epochs: {len(self.train_losses)}', fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Performance Summary')
        
        plt.tight_layout()
        plt.savefig('improved_model_training_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main training function for improved model"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
    
    # Define visit paths
    visit_paths = [
        "/neosp/tops/prod/s0012/t1/q3/l0/v00",
        "/neosp/tops/prod/s0012/t1/q3/l0/v01", 
        "/neosp/tops/prod/s0012/t1/q3/l0/v03"
    ]
    
    print(f"\nLoading data from {len(visit_paths)} visits:")
    for path in visit_paths:
        print(f"   - {path}")
    
    # Create model and dataset
    print(f"\nCreating foundation model with dual-channel ResNet + 50 features...")
    dataset = NEOSDataset(visit_paths)
    model = NEOSFoundationModel(tabular_dim=50, embed_dim=512)
    criterion = ContrastiveLoss()
    
    print(f"\nDataset Summary:")
    print(f"   Total samples: {len(dataset):,}")
    print(f"   Visits: {len(dataset.visit_names)}")
    print(f"   Visit names: {dataset.visit_names}")
    print(f"   Features: {len(dataset.selected_features)} (ALL selected)")
    
    print(f"\nModel Summary:")
    print(f"   Architecture: Dual-Channel ResNet + Tabular")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Image processing: NC1+NC2 → Single ResNet")
    print(f"   Tabular features: 50 normalized features")
    
    # Create trainer
    trainer = FoundationModelTrainer(
        model=model,
        dataset=dataset,
        criterion=criterion,
        device=device,
        batch_size=128,
        learning_rate=1e-4,
        weight_decay=1e-4
    )
    
    # Train the model
    print(f"\nStarting foundation model training...")
    train_losses = trainer.train(num_epochs=25)
    
    # Plot comprehensive analysis
    trainer.plot_training_analysis()
    
    print(f"\nFoundation model training completed successfully!")
    print(f"Final loss: {train_losses[-1]:.4f}")
    print(f"Best loss: {min(train_losses):.4f}")
    print(f"Trained on: {len(dataset):,} samples")
    print(f"Architecture: Dual-channel ResNet + 50 features")


if __name__ == "__main__":
    main()
