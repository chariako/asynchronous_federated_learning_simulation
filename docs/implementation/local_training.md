# Local Training and Aggregation Weights

Starting with a primer on Empirical Risk Minimization (ERM), this section demonstrates how aggregation weights at the server can be exchanged with appropriate gradient scaling at the client-level, and how AFL-Sim utilizes this conversion in its implementation of local client training.

## Federated Learning as Empirical Risk Minimization

Consider the case of $N$ total clients, where client $i \in \{1,...,N\}$ possesses the private dataset $D_i$. Let $n_i$ be the number of samples in $D_i$ and $n := \sum_{i=1}^N n_i$ be the total number of samples.

The global Empirical Risk (ER) for this FL problem can be formulated as:

$$R(w) = \frac{1}{n} \sum_{i=1}^N \sum_{s \in D_i} l(w, s),$$

where $w$ is the vector of model parameters, and $l(w, \cdot)$ is the loss function.

Let $\hat{R}_i$ be the local ER at client $i$:

$$\hat{R}_i(w):= \frac{1}{n_i} \sum_{s \in D_i} l(w, s).$$

Using this, we can rewrite the global ER as:

$$R(w) = \sum_{i=1}^N w_i \hat{R}_i(w),$$

where $w_i := \frac{n_i}{n}$.

The factor $w_i$ is the **importance** of client $i$ in the global objective. In standard FL implementations, it is used by the server as the aggregation weight for client $i$'s updates.

## Exchanging Aggregation Weights with Gradient Scaling

Define a new scaled local objective $R_i$ as:

$$R_i(w) := w_i \hat{R}_i(w).$$

Substituting $R_i$ in the global ER, we recover the following standard composite objective that frequently appears in distributed optimization:

$$R(w) = \sum_{i=1}^N R_i(w).$$

Note that **all clients have equal importance in this objective**. Because the relative dataset sizes are absorbed by $R_i$, the server can employ simple, uniform aggregation weights for all client updates.

However, for the system to converge to a correct solution of the global ER, clients must now take local (stochastic) gradient steps on the functions $R_i$ instead of the standard functions $\hat{R}_i$. This can be achieved by scaling the raw (stochastic) gradients $\nabla \hat{R}_i$ natively returned by frameworks like PyTorch with the factor $w_i$ to recover the gradient $\nabla R_i$:

$$\nabla{R}_i(w) = w_i \nabla \hat{R}_i(w).$$

!!! note

    Scaling the local stochastic gradients is precisely how AFL-Sim implements local client training. By transferring the relative dataset scaling from the server (during aggregation) to the clients (during the gradient step), the server architecture is simplified and homogenized. **This design choice is important to keep in mind when implementing custom aggregation weighting schemes on top of AFL-Sim.**
