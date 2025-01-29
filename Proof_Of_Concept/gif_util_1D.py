import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import io
import imageio
import os


def get_all_data(function_list):
    all_values = torch.linspace(-5 * np.pi, 5 * np.pi, 100).unsqueeze(1)
    #all_next_values = torch.sin(all_values)
    all_next_values = all_values.clone()
    for function in function_list:
        all_next_values = function(all_values)
    # Create pairs (x, y)
    data = torch.stack((all_values, all_next_values), dim=1)

    # Shuffle the data
    shuffled_indices = torch.randperm(len(data))
    shuffled_data = data[shuffled_indices]

    # Determine split indices
    split_index = int(0.8 * len(shuffled_data))  # 80% for training

    # Split into train and test sets
    train_data = shuffled_data[:split_index]
    test_data = shuffled_data[split_index:]

    # Generate the data
    super_values = torch.linspace(-10 * np.pi, 10 * np.pi, 300).unsqueeze(1)
    all_next_super_values = super_values.clone()
    #all_next_super_values = torch.sin(super_values)
    for function in function_list:
        all_next_super_values = function(super_values)
    super_data = torch.cat((super_values, all_next_super_values), dim=1)

    # Split data into ranges
    range_1_mask = (super_values.squeeze() >= -5 * np.pi) & (super_values.squeeze() <= 5 * np.pi)
    range_2_mask = (super_values.squeeze() > 5 * np.pi) | (super_values.squeeze() < -5 * np.pi)

    range_1_values = super_values[range_1_mask]
    range_1_actual = all_next_super_values[range_1_mask]
    range_2_values = super_values[range_2_mask]
    range_2_actual = all_next_super_values[range_2_mask]

    return train_data, test_data, super_data, range_1_values, range_1_actual, range_2_values, range_2_actual, range_1_mask, range_2_mask

class LQR(nn.Module):
    def __init__(self, enc_dim):
        super(LQR, self).__init__()
        self.A = torch.nn.Parameter(torch.randn(enc_dim, enc_dim))
       
        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, enc_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(enc_dim//2, enc_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(enc_dim, enc_dim),
        )

        self.state_decoder = torch.nn.Sequential(
            torch.nn.Linear(enc_dim, enc_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(enc_dim//2, 1)
        )
    def forward (self,x):
        xx = self.state_encoder(x)
        x_prime_prediction = self.A @ xx 
        return self.state_decoder(x_prime_prediction), x_prime_prediction, xx, #reward.unsqueeze(0)

#train, test are lists of tuples of (x, u, y, r)
def train_model(model,optimizer,train_data,epochs=1):
    criterion = torch.nn.MSELoss()
    for i in range(epochs):
        total_state_loss = 0
        total_reward_loss = 0
        for x, y in train_data:
            optimizer.zero_grad()
            lqr_x_prime, x_prime_expanded, xx = model(x)
            #reward_loss = criterion(reward, r)
            lqr_pred_loss = criterion(lqr_x_prime, y)
            decoder_loss = criterion(model.state_decoder(xx), x)
            encoder_loss = criterion(model.state_encoder(y), x_prime_expanded) 
            state_loss = lqr_pred_loss  + decoder_loss + encoder_loss
            loss = state_loss #+ reward_loss
            loss.backward()
            optimizer.step()
            total_state_loss += state_loss.item()
            total_reward_loss += 0#reward_loss.item()



def visualize_model_performance(model,embed_dim, epoch, file_name, super_data,range_1_values, range_1_actual, range_2_values, range_2_actual, 
                                range_1_mask, range_2_mask):
    test_predictions = []
    with torch.no_grad():
        for x, y in super_data:
            lqr_x_prime, x_prime_expanded, xx = model(x.unsqueeze(0))
            test_predictions.append(lqr_x_prime)
    test_predictions = torch.tensor(test_predictions)

    # Scatter plot
    plt.figure()
    plt.scatter(
        range_1_values, 
        test_predictions[range_1_mask], 
        c='r', alpha=0.5, label="Predicted (Training Range)"
    )
    plt.scatter(
        range_1_values, 
        range_1_actual, 
        c='b', alpha=0.5, label="Actual (Training Range)"
    )
    plt.scatter(
        range_2_values, 
        test_predictions[range_2_mask], 
        c='orange', alpha=0.5, label="Predicted (Unseen Range)"
    )
    plt.scatter(
        range_2_values, 
        range_2_actual, 
        c='green', alpha=0.5, label="Actual (Unseen Range)"
    )

    plt.title(f'{file_name}, Dim: {embed_dim} Epoch: {epoch}')
    plt.xlabel('Neg 10 pi to 10 pi')
    #plt.ylabel('Sin(Pi)')
    #plt.ylim(-1.2, 2)
    plt.xlim(-10 * np.pi, 10 * np.pi)

    # Save plot as an in-memory object
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf  # Return the buffer containing the image


def make_gif(model, embed_dim,num_epochs, super_data, train_data, file_name, range_1_values, range_1_actual, 
             range_2_values, range_2_actual, range_1_mask, range_2_mask):
    images = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, num_epochs + 1):
        train_model(model, optimizer, train_data)
        img_buf = visualize_model_performance(
            model,embed_dim, epoch, file_name, super_data, range_1_values, range_1_actual, range_2_values, range_2_actual, range_1_mask, range_2_mask
        )
        images.append(imageio.imread(img_buf))
        img_buf.close()

    # Create GIF
    os.makedirs('gifs', exist_ok=True)
    imageio.mimsave(f'gifs/{embed_dim}_{file_name}.gif', images)  # Adjust duration as needed
