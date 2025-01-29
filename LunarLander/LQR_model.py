import torch
import torch.nn as nn

class LQRModel(nn.Module):
    def __init__(self, hidden_state_dim=64, hidden_action_dim=64, dropout_rate=0.3):
        super(LQRModel, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(8, hidden_state_dim),
            nn.ReLU(),
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.ReLU(),
            nn.Linear(hidden_state_dim, hidden_state_dim),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(2, hidden_action_dim),
            nn.ReLU(),
            nn.Linear(hidden_action_dim, hidden_action_dim),
            nn.ReLU(),
            nn.Linear(hidden_action_dim, hidden_action_dim),
        )

        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.ReLU(),
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.ReLU(),
            nn.Linear(hidden_state_dim, 8)
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_action_dim, hidden_action_dim),
            nn.ReLU(),
            nn.Linear(hidden_action_dim, hidden_action_dim),
            nn.ReLU(),
            nn.Linear(hidden_action_dim, 2)
        )

        self.A = nn.Parameter(torch.randn(hidden_state_dim, hidden_state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(hidden_state_dim, hidden_action_dim) * 0.01)

        #Q_base and R_base are "base" since they become Q and R after making them positive semi-definite
        self.Q_base = nn.Parameter(torch.randn(hidden_state_dim, hidden_state_dim))
        self.R_base = nn.Parameter(torch.randn(hidden_action_dim, hidden_action_dim))

    def get_positive_semi_definite_matrix(self, base_matrix):
        return base_matrix.T @ base_matrix + torch.eye(base_matrix.size(0)) * 1e-6
#Input: Bx8 State, Bx2 Action
    def forward(self, state, action):
        batch_size = state.size(0)
        enc_state = self.state_encoder(state).unsqueeze(-1)
        enc_action = self.action_encoder(action).unsqueeze(-1)


        A_batch = self.A.unsqueeze(0).expand(batch_size, -1, -1)
        #A_batch is a Bx64x64 matrix
        # Add a small regularization term to the diagonal for numerical stability
        A_batch = A_batch + 1e-6 * torch.eye(self.A.size(0)).unsqueeze(0).expand(batch_size, -1, -1)
        B_batch = self.B.unsqueeze(0).expand(batch_size, -1, -1)

        #x_dot is the predicted delta in state, not the next state
        x_dot_pred = torch.bmm(A_batch, enc_state) + torch.bmm(B_batch, enc_action)        
        x_prime = enc_state.squeeze(2) + x_dot_pred.squeeze(2)

        #Bx64x1 -> Bx64
        Q = self.get_positive_semi_definite_matrix(self.Q_base) 
        R = self.get_positive_semi_definite_matrix(self.R_base)

        Q_batch = Q.unsqueeze(0).expand(batch_size, -1, -1)
        R_batch = R.unsqueeze(0).expand(batch_size, -1, -1)

        #state_cost = state_enc^T Q state_enc
        state_cost = torch.bmm(enc_state.transpose(1, 2), torch.bmm(Q_batch, enc_state)).squeeze()
        action_cost = torch.bmm(enc_action.transpose(1, 2), torch.bmm(R_batch, enc_action)).squeeze()

        reward_pred = -(state_cost + action_cost)

        return x_prime, reward_pred