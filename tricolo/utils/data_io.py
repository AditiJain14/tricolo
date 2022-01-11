import os 

def get_voxel_file(dataset, category, model_id):
    """Get the voxel absolute filepath for the model specified by category
    and model_id.
    Args:
        category: Category of the model as a string (eg. '03001627')
        model_id: Model ID of the model as a string
            (eg. '587ee5822bb56bd07b11ae648ea92233')
    Returns:
        voxel_filepath: Filepath of the binvox file corresonding to category and
            model_id.
    """
    if dataset == 'shapenet':  # ShapeNet dataset
        return os.path.join("../data/retrieval/shapenet/nrrd_256_filter_div_64_solid", model_id, model_id+'.nrrd')
    elif dataset == 'primitives':  # Primitives dataset
        return os.path.join("../data/retrieval/primitives/nrrd_32_solid", category, model_id+'.nrrd')
    else:
        raise ValueError('Please choose a valid dataset (shapenet, primitives).')
