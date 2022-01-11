import argparse
import collections
import datetime
import json
import numpy as np
import os
import pickle

import jsonlines


def construct_embeddings_matrix(dataset, embeddings_dict, model_id_to_label=None,
                                label_to_model_id=None):
    """Construct the embeddings matrix, which is NxD where N is the number of embeddings and D is
    the dimensionality of each embedding.
    Args:
        dataset: String specifying the dataset (e.g. 'synthetic' or 'shapenet')
        embeddings_dict: Dictionary containing the embeddings. It should have keys such as
                the following: ['caption_embedding_tuples', 'dataset_size'].
                caption_embedding_tuples is a list of tuples where each tuple can be decoded like
                so: caption, category, model_id, embedding = caption_tuple.
    """
    assert (((model_id_to_label is None) and (label_to_model_id is None)) or
            ((model_id_to_label is not None) and (label_to_model_id is not None)))
    embedding_sample = embeddings_dict['caption_embedding_tuples'][0][-1]
    embedding_dim = embedding_sample.shape[0]
    num_embeddings = len(embeddings_dict['caption_embedding_tuples'])

    assert embedding_sample.ndim == 1

    # Create embeddings matrix (n_samples x n_features) and vector of labels
    text_embeddings_matrix = np.zeros((num_embeddings, embedding_dim))
    shape_embeddings_list = []
    labels = np.zeros((num_embeddings)).astype(int)
    labels_shape = []

    if (model_id_to_label is None) and (label_to_model_id is None):
        model_id_to_label = {}
        label_to_model_id = {}
        label_counter = 0
        new_dicts = True
    else:
        new_dicts = False

    for idx, caption_tuple in enumerate(embeddings_dict['caption_embedding_tuples']):

        # Parse caption tuple
        caption, category, model_id, text_embedding, shape_embedding = caption_tuple

        # Swap model ID and category depending on dataset
        if dataset == 'primitives':
            tmp = model_id
            model_id = category
            category = tmp

        # Add model ID to dict if it has not already been added
        if new_dicts:
            if model_id not in model_id_to_label:
                model_id_to_label[model_id] = label_counter
                label_to_model_id[label_counter] = model_id
                label_counter += 1

                shape_embeddings_list.append(shape_embedding)
                labels_shape.append(label_counter-1)

        # Update the embeddings matrix and labels vector
        text_embeddings_matrix[idx] = text_embedding
        labels[idx] = model_id_to_label[model_id]

        # Print progress
        if (idx + 1) % 10000 == 0:
            print('Processed {} / {} embeddings'.format(idx + 1, num_embeddings))

    shape_embeddings_matrix = np.vstack(shape_embeddings_list)
    labels_shape = np.array(labels_shape).astype(int)
    num_embeddings = len(shape_embeddings_list)
    return text_embeddings_matrix, shape_embeddings_matrix, labels, labels_shape, model_id_to_label, num_embeddings, label_to_model_id


def print_model_id_info(model_id_to_label):
    print('Number of models (or categories if synthetic dataset):', len(model_id_to_label.keys()))
    print('')


def _compute_nearest_neighbors_cosine(fit_embeddings_matrix, query_embeddings_matrix,
                                      n_neighbors, fit_eq_query, range_start=0):
    if fit_eq_query is True:
        n_neighbors += 1

    # Argsort method
    unnormalized_similarities = np.dot(query_embeddings_matrix, fit_embeddings_matrix.T)
    sort_indices = np.argsort(unnormalized_similarities, axis=1)
    sort_distances = np.sort(unnormalized_similarities, axis=1)
    distances = sort_distances[:, -n_neighbors:]
    distances = np.flip(distances)
    # return unnormalized_similarities[:, -n_neighbors:], sort_indices[:, -n_neighbors:]
    indices = sort_indices[:, -n_neighbors:]
    indices = np.flip(indices, 1)
    sort_indices = np.flip(sort_indices, 1)

    if fit_eq_query is True:
        n_neighbors -= 1  # Undo the neighbor increment
        final_indices = np.zeros((indices.shape[0], n_neighbors), dtype=int)
        compare_mat = np.asarray(list(range(range_start, range_start + indices.shape[0]))).reshape(indices.shape[0], 1)
        has_self = np.equal(compare_mat, indices)  # has self as nearest neighbor
        any_result = np.any(has_self, axis=1)
        for row_idx in range(indices.shape[0]):
            if any_result[row_idx]:
                nonzero_idx = np.nonzero(has_self[row_idx, :])
                assert len(nonzero_idx) == 1
                new_row = np.delete(indices[row_idx, :], nonzero_idx[0])
                final_indices[row_idx, :] = new_row
            else:
                final_indices[row_idx, :] = indices[row_idx, :n_neighbors]
        indices = final_indices
    return distances, indices, sort_indices


def compute_nearest_neighbors_cosine(fit_embeddings_matrix, query_embeddings_matrix,
                                     n_neighbors, fit_eq_query):
    # print('Using normalized cosine distance')
    n_samples = query_embeddings_matrix.shape[0]
    if n_samples > 8000:  # Divide into blocks and execute
        def block_generator(mat, block_size):
            for i in range(0, mat.shape[0], block_size):
                yield mat[i:(i + block_size), :]

        block_size = 3000
        blocks = block_generator(query_embeddings_matrix, block_size)
        indices_list = []
        distances_list, sort_indices_list = [], []
        for cur_block_idx, block in enumerate(blocks):
            print('Nearest neighbors on block {}'.format(cur_block_idx + 1))
            cur_distances, cur_indices, cur_sort_indices = _compute_nearest_neighbors_cosine(fit_embeddings_matrix, block,
                                                            n_neighbors, fit_eq_query,
                                                            range_start=cur_block_idx * block_size)
            indices_list.append(cur_indices)
            distances_list.append(cur_distances)
            sort_indices_list.append(cur_sort_indices)
        indices = np.vstack(indices_list)
        distances = np.vstack(distances_list)
        sort_indices = np.vstack(sort_indices_list)
        return distances, indices, sort_indices
    else:
        distances, indices, sort_indices = _compute_nearest_neighbors_cosine(fit_embeddings_matrix,
                                                       query_embeddings_matrix, n_neighbors,
                                                       fit_eq_query)
        return distances, indices, sort_indices


def compute_nearest_neighbors(fit_embeddings_matrix, query_embeddings_matrix,
                              n_neighbors, metric='cosine'):
    """Compute nearest neighbors.
    Args:
        fit_embeddings_matrix: NxD matrix
    """
    fit_eq_query = False
    if ((fit_embeddings_matrix.shape == query_embeddings_matrix.shape)
        and np.allclose(fit_embeddings_matrix, query_embeddings_matrix)):
        fit_eq_query = True

    if metric == 'cosine':
        distances, indices, sort_indices = compute_nearest_neighbors_cosine(fit_embeddings_matrix,
                                                              query_embeddings_matrix,
                                                              n_neighbors, fit_eq_query)
    else:
        raise ValueError('Use cosine distance.')
    return distances, indices, sort_indices


def compute_pr_at_k(embeddings_path, indices, sort_indices, labels, n_neighbors, num_embeddings, fit_labels=None):
    """Compute precision and recall at k (for k=1 to n_neighbors)
    Args:
        indices: num_embeddings x n_neighbors array with ith entry holding nearest neighbors of
                 query i
        labels: 1-d array with correct class of query
        n_neighbors: number of neighbors to consider
        num_embeddings: number of queries
    """
    fo = open(os.path.join(embeddings_path, "pr_at_k.txt"), 'w')

    if fit_labels is None:
        fit_labels = labels
    num_correct = np.zeros((num_embeddings, n_neighbors))
    rel_score = np.zeros((num_embeddings, n_neighbors))
    label_counter = np.bincount(fit_labels)
    num_relevant = label_counter[labels]
    rel_score_ideal = np.zeros((num_embeddings, n_neighbors))

    # Assumes that self is not included in the nearest neighbors
    r_rank = 0
    for i in range(num_embeddings):
        label = labels[i]  # Correct class of the query
        nearest = indices[i]  # Indices of nearest neighbors
        nearest_classes = [fit_labels[x] for x in nearest]  # Class labels of the nearest neighbors
        # for now binary relevance
        num_relevant_clamped = min(num_relevant[i], n_neighbors)
        rel_score[i] = np.equal(np.asarray(nearest_classes), label)
        rel_score_ideal[i][0:num_relevant_clamped] = 1

        for k in range(n_neighbors):
            # k goes from 0 to n_neighbors-1
            correct_indicator = np.equal(np.asarray(nearest_classes[0:(k + 1)]), label)  # Get true (binary) labels
            num_correct[i, k] = np.sum(correct_indicator)

        # r_rank
        all_nearest = sort_indices[i]
        all_nearest_classes = [fit_labels[x] for x in all_nearest]
        r_rank += 1 / (all_nearest_classes.index(label) + 1)
    r_rank = r_rank / num_embeddings

    # Compute our dcg
    dcg_n = np.exp2(rel_score) - 1
    dcg_d = np.log2(np.arange(1,n_neighbors+1)+1)
    dcg = np.cumsum(dcg_n/dcg_d,axis=1)
    # Compute ideal dcg
    dcg_n_ideal = np.exp2(rel_score_ideal) - 1
    dcg_ideal = np.cumsum(dcg_n_ideal/dcg_d,axis=1)
    # Compute ndcg
    ndcg = dcg / dcg_ideal
    ave_ndcg_at_k = np.sum(ndcg, axis=0) / num_embeddings
    recall_rate_at_k = np.sum(num_correct > 0, axis=0) / num_embeddings
    recall_at_k = np.sum(num_correct/num_relevant[:,None], axis=0) / num_embeddings
    precision_at_k = np.sum(num_correct/np.arange(1,n_neighbors+1), axis=0) / num_embeddings
    #print('recall_at_k shape:', recall_at_k.shape)
    print('     k: precision recall recall_rate ndcg r_rank')
    fo.write('     k: precision recall recall_rate ndcg r_rank\n')
    for k in range(n_neighbors):
        print('pr @ {}: {} {} {} {} {}'.format(k + 1, precision_at_k[k], recall_at_k[k], recall_rate_at_k[k], ave_ndcg_at_k[k], r_rank))
        fo.write('pr @ {}: {} {} {} {} {}'.format(k + 1, precision_at_k[k], recall_at_k[k], recall_rate_at_k[k], ave_ndcg_at_k[k], r_rank) + '\n')
    Metrics = collections.namedtuple('Metrics', 'precision recall recall_rate ndcg r_rank')

    fo.close()

    return Metrics(precision_at_k, recall_at_k, recall_rate_at_k, ave_ndcg_at_k, r_rank)


def get_nearest_info(indices, sort_indices, fit_labels, labels, label_to_model_id, caption_tuples, idx_to_word):
    """Compute and return the model IDs of the nearest neighbors.
    """
    # r_rank
    r_rank_list = [] 
    for i in range(len(sort_indices)):
        label = labels[i]
        all_nearest = sort_indices[i]
        all_nearest_classes = [fit_labels[x] for x in all_nearest]
        r_rank = 1 / (all_nearest_classes.index(label) + 1)
        r_rank_list.append(r_rank)

    # Convert labels to model IDs
    query_model_ids = []
    for idx, label in enumerate(labels):
        query_model_ids.append(label_to_model_id[label])

    # Convert neighbors to model IDs
    nearest_sentences = []
    nearest_model_ids = []
    for row in indices:
        cur_nearest_sentences = []
        cur_nearest_model_ids = []
        for col in row:
            cur_nearest_model_ids.append(caption_tuples[col][2])
            cur_sentence_as_word_indices = caption_tuples[col][0]
            cur_nearest_sentences.append(' '.join([idx_to_word[str(word_idx)]
                                                for word_idx in cur_sentence_as_word_indices
                                                if word_idx != 0]))
        nearest_sentences.append(cur_nearest_sentences)
        nearest_model_ids.append(cur_nearest_model_ids)    

    return query_model_ids, nearest_model_ids, nearest_sentences, r_rank_list


def print_nearest_info(dataset, query_model_ids, nearest_model_ids, nearest_sentences, r_ranks, distances,
                       render_dir_input):
    """Print out nearest model IDs for random queries.
    Args:
        labels: 1D array containing the label
    """
    # Make directory for renders
    render_dir = os.path.join(render_dir_input, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(render_dir)
    print("render dir is ", render_dir)

    num_queries = 50
    # perm = np.random.permutation(len(nearest_model_ids))
    cnt = 0

    jsonl_writer = jsonlines.open(os.path.join(render_dir,'nearest.jsonl'), mode='w')

    # for i in perm[:num_queries]:
    for i in range(len(nearest_sentences)):
        query_model_id = query_model_ids[i]
        distance = distances[i]
        nearest_mod = nearest_model_ids[i]
        nearest_sen = nearest_sentences[i]
        r_rank = r_ranks[i]
        if r_rank < 0.2:
            continue

        # Make directory for the query
        cur_render_dir = os.path.join(render_dir, query_model_id + ('-%04d' % i))
        os.makedirs(cur_render_dir)

        jsonl_writer.write({'groundtruth': query_model_id + ('-%04d' % i), 'retrieval_modelids': nearest_mod, 'retrieved_sentences': nearest_sen, 'distance': distance.tolist()})

        with open(os.path.join(cur_render_dir, 'nearest_neighbor_text.txt'), 'w') as f:
            f.write('Reciprocal Rank: {}\n'.format(r_rank))

            f.write('-------- query {} ----------\n'.format(i))
            f.write('Query: {}\n'.format(query_model_id))
            f.write('Nearest:\n')
            for model_id in nearest_mod:
                f.write('\t{}\n'.format(model_id))
            for sen in nearest_sen:
                f.write('\t{}\n'.format(sen))
            
        cnt += 1
        if cnt >= num_queries:
            break

def compute_metrics(info, dataset, embeddings_dict, embeddings_path, metric='cosine', concise=False):
    """Compute all the metrics for the text encoder evaluation.
    """
    (text_embeddings_matrix, shape_embeddings_matrix, labels, fit_labels, model_id_to_label,
     num_embeddings, label_to_model_id) = construct_embeddings_matrix(
        dataset,
        embeddings_dict
    )

    print_model_id_info(model_id_to_label)

    n_neighbors = 5 # 20 change

    distances, indices, sort_indices = compute_nearest_neighbors(text_embeddings_matrix, shape_embeddings_matrix, n_neighbors, metric=metric) # change text_em and shape_em

    print('Computing precision recall.')
    pr_at_k = compute_pr_at_k(embeddings_path, indices, sort_indices, fit_labels, n_neighbors, num_embeddings, labels) # change labels and fit_labels
    
    if concise is False or isinstance(concise, str):
        with open(info, 'r') as f:
            inputs_list = json.load(f)
        idx_to_word = inputs_list['idx_to_word']

        query_model_ids, nearest_model_ids, nearest_sentences, r_ranks = get_nearest_info(
            indices,
            sort_indices,
            labels,
            fit_labels,
            label_to_model_id,
            embeddings_dict['caption_embedding_tuples'],
            idx_to_word,
        ) # change labels and fit_labels

        out_dir = concise if isinstance(concise, str) else None
        print_nearest_info(dataset, query_model_ids, nearest_model_ids, nearest_sentences, r_ranks, distances,
                           render_dir_input=out_dir)

    return pr_at_k
    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', help='dataset (''shapenet'', ''primitives'')', default='shapenet')
    parser.add_argument('--embeddings_path', help='path to text embeddings pickle file', default='logs/retrieval/v64i128b128/Nov12_05-26-10-2/test/output.p')
    parser.add_argument('--metric', help='path to text embeddings pickle file', default='cosine',
                        type=str)
    args = parser.parse_args()

    with open(args.embeddings_path, 'rb') as f:
        embeddings_dict = pickle.load(f)

    render_dir = os.path.join(os.path.dirname(args.embeddings_path), 'nearest_neighbor_renderings')
    np.random.seed(1234)
    info = "../data/text2shape-data/shapenet/shapenet.json"
    compute_metrics(info, args.dataset, embeddings_dict, os.path.dirname(args.embeddings_path), metric=args.metric, concise=render_dir)


if __name__ == '__main__':
    main()