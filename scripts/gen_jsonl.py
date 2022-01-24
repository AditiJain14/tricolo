'''
This script is for shapenet-related data process
'''
import jsonlines
import pickle

# jsonl, model, category, caption, arrays

def create_text_json():
    '''
    find modelid-caption pairs 
    store them in train_map.json
    '''
    
    # with jsonlines.open("../data/train_map.jsonl") as reader:
    #        clr_frame = list(reader)

    ############ train split ####################
    with open('../data/text2shape-data/shapenet/processed_captions_train.p', 'rb') as f: # change
        train_data = pickle.load(f)
    matches = train_data["caption_matches"]
    tuples = train_data["caption_tuples"]

    train_ids = matches.keys()
    print(f"how many train modelids: {len(train_ids)}") # train: 11921, val:1486 test: 1492, 8:1:1
    
    target_list = []
    for k in matches:
        all_arrays = []
        for ind in matches[k]:
            arrays = tuples[ind][0].tolist()
            all_arrays.append(arrays)
        
        for ind in matches[k]:
            arrays = tuples[ind][0].tolist()
            category = tuples[ind][1]
            modelid = tuples[ind][2]
            target_list.append({"model": modelid, "category": category, "arrays": arrays, "all_arrays": all_arrays})
    
    with jsonlines.open('../data/text2shape-data/shapenet/train_map.jsonl', mode='w') as writer:
        writer.write_all(target_list)
            
    ############ validation split ####################
    with open('../data/text2shape-data/shapenet/processed_captions_val.p', 'rb') as f: # change
        val_data = pickle.load(f)
    matches = val_data["caption_matches"]
    tuples = val_data["caption_tuples"]

    val_ids = matches.keys()
    print(f"how many train modelids: {len(val_ids)}") # train: 11921, val:1486 test: 1492
    
    target_list = []
    for k in matches:
        all_arrays = []
        for ind in matches[k]:
            arrays = tuples[ind][0].tolist()
            all_arrays.append(arrays)
        
        for ind in matches[k]:
            arrays = tuples[ind][0].tolist()
            category = tuples[ind][1]
            modelid = tuples[ind][2]
            target_list.append({"model": modelid, "category": category, "arrays": arrays, "all_arrays": all_arrays})
    
    with jsonlines.open('../data/text2shape-data/shapenet/val_map.jsonl', mode='w') as writer:
        writer.write_all(target_list)
            
            
    ############ test split ####################
    with open('../data/text2shape-data/shapenet/processed_captions_test.p', 'rb') as f: # change
        test_data = pickle.load(f)
    matches = test_data["caption_matches"]
    tuples = test_data["caption_tuples"]

    test_ids = matches.keys()
    print(f"how many test modelids: {len(test_ids)}") # train: 11921, val:1486 test: 1492
    
    target_list = []
    for k in matches:
        all_arrays = []
        for ind in matches[k]:
            arrays = tuples[ind][0].tolist()
            all_arrays.append(arrays)
        
        for ind in matches[k]:
            arrays = tuples[ind][0].tolist()
            category = tuples[ind][1]
            modelid = tuples[ind][2]
            target_list.append({"model": modelid, "category": category, "arrays": arrays, "all_arrays": all_arrays})
    
    with jsonlines.open('../data/text2shape-data/shapenet/test_map.jsonl', mode='w') as writer:
        writer.write_all(target_list)
            
    



if __name__=='__main__':
    create_text_json() 
