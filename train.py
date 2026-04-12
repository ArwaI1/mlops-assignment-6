import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")
    df = pd.read_csv(path + "/sign_mnist_train.csv")

    y = df['label'].values
    X = df.drop('label', axis=1).values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape and Normalize (Adjusted to PyTorch standard: N, C, H, W)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_train = (X_train - 127.5) / 127.5 
    
    print(f"Data processed! Training shape: {X_train.shape}")
    return X_train, y_train

def main():
    # Setup terminal arguments for easy testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--run_name", type=str, default="exp1") 
    args = parser.parse_args()

    # CONNECT TO YOUR LOCAL SERVER (Fix for the missing runs in UI)
    mlflow.set_tracking_uri("http://localhost:5000")

    # MLFLOW PILLAR 1: Set Experiment Name
    mlflow.set_experiment("A3_Arwa Mater")

    # Load and preprocess Kaggle dataset
    X_train, y_train = load_and_preprocess_data()
    
    # Convert numpy arrays to PyTorch Tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    
    # Create dataset and dataloader
    train_data = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # Simplified Model (No classes used)
    # Output is 25 because Sign Language MNIST labels range from 0 to 24
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 25) 
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    # MLFLOW PILLAR 2: Wrap training in start_run() AND pass the run_name!
    with mlflow.start_run(run_name=args.run_name):
        
        # MLFLOW PILLAR 3: Log Parameters
        mlflow.log_params({
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs
        })
        
        # MLFLOW PILLAR 4: Add Tags
        mlflow.set_tag("student_id", "202201623")

        print(f"Training Run '{args.run_name}' with LR: {args.learning_rate}, Batch: {args.batch_size}")

        for epoch in range(args.epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total

            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

            # MLFLOW PILLAR 5: Live Logging
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)

        # MLFLOW PILLAR 6: Save Model Flavor
        mlflow.pytorch.log_model(model, "model")
        print("Run complete and saved to MLflow!")

if __name__ == "__main__":
    main()