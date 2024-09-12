import torch
from train import Transformer, decode  
from datetime import datetime
import os 


def gen(MODEL, text_length=10000, terminal=True, writefile=True, timestamp=True):
    device = "cuda" if torch.cuda.is_available() else ("cpu" if torch.backends.mps.is_available() else "cpu")
    model = Transformer()

    # Initialize the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"models/{MODEL}.pth")

    model.load_state_dict(torch.load(file_path, weights_only=True, map_location=device))  # Load model's state dict
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    print("-=-=-=-=-= model starting .. =-=-=-=-=-")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Assuming single batch, single token start

    gen_text = decode(model.generate(context, max_new_tokens=text_length)[0].tolist())  # Generate new tokens and decode
    print("-=-=-=-=-= context generated =-=-=-=-=-")
    
    if terminal: print(gen)

    if writefile:
        appendix = timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if timestamp else ""
        file_path = os.path.join(current_dir, f"generated/out_{appendix}.txt")

        os.makedirs("generated", exist_ok=True)  # Create directory if it doesn't exist
        with open(file_path, "w") as f:
            f.write(gen_text)
        print(f"File saved at: generated/out_{appendix}.txt")


if __name__ == "__main__":
    gen(MODEL="1984")
