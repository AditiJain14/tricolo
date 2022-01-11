import argparse
import jsonlines
import os
import re
import shutil

OUTPUT_FILE = "retrieval_results.html"

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, help="Path to the result folder containing nearest.jsonl and images", default='logs/retrieval/Nov09_21-45-44_Cfg0_GPU0/predict/nearest_neighbor_renderings/2021-11-11_16-33-27')
args = parser.parse_args()

def read_data():
    
    info = os.path.join(args.root_dir, 'nearest.jsonl')
    with jsonlines.open(info) as records: 
        results = list(records)   
    return results

def get_pure_caption(caption):
    words = caption.split(" ")
    for i in range(len(words)):
        if words[i] == "[SEP]":
            break
    return " ".join(words[:i+1])

def image_html(shape_dir, model_id):
    image_path = os.path.join('../data/retrieval/shapenet/nrrd_256_filter_div_64_solid', model_id, model_id+'.png')
    assert os.path.exists(image_path), f"picture doesn't exist! {image_path}"
    destination = os.path.join(args.root_dir, shape_dir, model_id+'.png')
    shutil.copy(image_path, destination)
    return '<img src="{}/{}.png" alt="{}" width="80" height="100">'.format(shape_dir, model_id, model_id)

def generate_html(results):
    html = "<h1>Shape Retrieval Results"
    i = 0
    all_folder = os.listdir(args.root_dir)
    for result in results:
        p = re.compile(result["groundtruth"])
        shape_dir = [folder for folder in all_folder if p.match(folder) is not None]
        result_html = "<h2>Result {}</h2>caption: <b>{}</b><br><br>ground_truth: <br>{}<br>retrieved models: <br>".format(i, get_pure_caption(result["caption"]), image_html(shape_dir[0], result["groundtruth"].split('-')[0]))


        for model_id in result["retrieved_models"]:
            result_html += image_html(shape_dir[0], model_id)
        html += result_html + "<br><br><br>\n\n"
        i += 1
    return html

def write_data(html):
    f = open(os.path.join(args.root_dir, OUTPUT_FILE), "w")
    f.write(html)
    f.close()

def main():
    write_data(generate_html(read_data()))

if __name__ == "__main__":
    main()

