import numpy as np
import pandas as pd
import random
import scipy.sparse as sp
import tensorflow as tf

from tensorflow.keras.utils import Progbar


class GraphConv(tf.keras.layers.Layer):
    def __init__(self, adj_mat, weight=True):
        super(GraphConv, self).__init__()
        self.adj_mat = adj_mat
        self.weight = weight

    def build(self, input_shape):
        if self.weight:
            self.W = self.add_weight('kernel',
                                     shape=[int(input_shape[-1]),
                                            int(input_shape[-1])])

    def call(self, ego_embeddings):
        output = tf.sparse.sparse_dense_matmul(self.adj_mat, ego_embeddings)
        if self.weight:
            output = tf.transpose(tf.matmul(self.W, output, transpose_a=False, transpose_b=True))
        return output


class LightGCN(tf.keras.Model):
    def __init__(self, adj_mat, n_users, n_items, n_layers=3, emb_dim=64):
        super(LightGCN, self).__init__()
        self.adj_mat = adj_mat
        self.R = tf.sparse.to_dense(adj_mat)[:n_users, n_users:]
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.emb_dim = emb_dim

        # Initialize user and item embeddings.
        initializer = tf.keras.initializers.GlorotNormal()
        self.user_embedding = tf.Variable(
            initializer([self.n_users, self.emb_dim]), name='user_embedding'
        )
        self.item_embedding = tf.Variable(
            initializer([self.n_items, self.emb_dim]), name='item_embedding'
        )

        # Stack light graph convolutional layers.
        self.gcn = [GraphConv(adj_mat, weight=False) for layer in range(n_layers)]

    def call(self, inputs):
        user_emb, item_emb = inputs
        output_embeddings = tf.concat([user_emb, item_emb], axis=0)
        all_embeddings = [output_embeddings]

        # Graph convolutions.
        for i in range(0, self.n_layers):
            output_embeddings = self.gcn[i](output_embeddings)
            all_embeddings += [output_embeddings]

        # Compute the mean of all layers
        all_embeddings = tf.stack(all_embeddings, axis=1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)

        # Split into users and items embeddings
        new_user_embeddings, new_item_embeddings = tf.split(
            all_embeddings, [self.n_users, self.n_items], axis=0
        )

        return new_user_embeddings, new_item_embeddings

    def fit(self, epochs=10, batch_size=128, optimizer=None, decay=0.0001):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Keep track of which movies each user has reviewed.
        interacted = (
            pd.DataFrame(
                {"userId_new": np.nonzero(self.R)[0], "movie_interacted": np.nonzero(self.R)[1]}
            )
                .groupby("userId_new")["movie_interacted"]
                .apply(set)
                .reset_index()
        )

        # Custom training loop from scratch.
        for epoch in range(1, epochs + 1):
            print('Epoch %d/%d' % (epoch, epochs))
            n_batch = tf.math.count_nonzero(self.R).numpy() // batch_size + 1
            bar = Progbar(n_batch, stateful_metrics='training loss')
            for idx in range(1, n_batch + 1):
                # Sample batch_size number of users with positive and negative items.
                indices = range(self.n_users)
                if self.n_users < batch_size:
                    users = np.array([random.choice(indices) for _ in range(batch_size)])
                else:
                    users = np.array(random.sample(indices, batch_size))

                def sample_neg(x):
                    while True:
                        neg_id = random.randint(0, self.n_items - 1)
                        if neg_id not in x:
                            return neg_id

                # Sample a single movie for each user that the user did and did not review.
                interact = interacted.iloc[users]
                pos_items = interact['movie_interacted'].apply(lambda x: random.choice(list(x)))
                neg_items = interact['movie_interacted'].apply(lambda x: sample_neg(x))

                users, pos_items, neg_items = users, np.array(pos_items), np.array(neg_items)

                with tf.GradientTape() as tape:
                    # Call model with user and item embeddings.
                    new_user_embeddings, new_item_embeddings = self(
                        (self.user_embedding, self.item_embedding)
                    )

                    # Embeddings after convolutions.
                    user_embeddings = tf.nn.embedding_lookup(new_user_embeddings, users)
                    pos_item_embeddings = tf.nn.embedding_lookup(new_item_embeddings, pos_items)
                    neg_item_embeddings = tf.nn.embedding_lookup(new_item_embeddings, neg_items)

                    # Initial embeddings before convolutions.
                    old_user_embeddings = tf.nn.embedding_lookup(
                        self.user_embedding, users
                    )
                    old_pos_item_embeddings = tf.nn.embedding_lookup(
                        self.item_embedding, pos_items
                    )
                    old_neg_item_embeddings = tf.nn.embedding_lookup(
                        self.item_embedding, neg_items
                    )

                    # Calculate loss.
                    pos_scores = tf.reduce_sum(
                        tf.multiply(user_embeddings, pos_item_embeddings), axis=1
                    )
                    neg_scores = tf.reduce_sum(
                        tf.multiply(user_embeddings, neg_item_embeddings), axis=1
                    )
                    regularizer = (
                            tf.nn.l2_loss(old_user_embeddings)
                            + tf.nn.l2_loss(old_pos_item_embeddings)
                            + tf.nn.l2_loss(old_neg_item_embeddings)
                    )
                    regularizer = regularizer / batch_size
                    mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
                    emb_loss = decay * regularizer
                    loss = mf_loss + emb_loss

                # Retreive and apply gradients.
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                bar.add(1, values=[('training loss', float(loss))])


    def recommend(self, users, k):
        # Calculate the scores.
        new_user_embed, new_item_embed = self((self.user_embedding, self.item_embedding))
        user_embed = tf.nn.embedding_lookup(new_user_embed, users)
        test_scores = tf.matmul(user_embed, new_item_embed, transpose_a=False, transpose_b=True)
        test_scores = np.array(test_scores)

        # Remove movies already seen.
        test_scores += sp.csr_matrix(self.R)[users, :] * -np.inf

        # Get top movies.
        test_user_idx = np.arange(test_scores.shape[0])[:, None]
        top_items = np.argpartition(test_scores, -k, axis=1)[:, -k:]
        top_scores = test_scores[test_user_idx, top_items]
        sort_ind = np.argsort(-top_scores)
        top_items = top_items[test_user_idx, sort_ind]
        top_scores = top_scores[test_user_idx, sort_ind]
        top_items, top_scores = np.array(top_items), np.array(top_scores)

        # Create Dataframe with recommended movies.
        topk_scores = pd.DataFrame(
            {
                'userId': np.repeat(users, top_items.shape[1]),
                'movieId': top_items.flatten(),
                'prediction': top_scores.flatten(),
            }
        )

        return topk_scores

class NGCF(LightGCN):
    def __init__(self, adj_mat, n_users, n_items, n_layers=3, emb_dim=64):
        super(NGCF, self).__init__(adj_mat, n_users, n_items, n_layers=3, emb_dim=64)

        # Stack graph convolutional layers.
        self.gcn = [GraphConv(adj_mat, weight=True) for layer in range(n_layers)]
