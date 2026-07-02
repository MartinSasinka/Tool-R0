import json, jsonlines
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
binarizer = MultiLabelBinarizer()

def read_jsonlines(path):
    with jsonlines.open(path, 'r') as file:
        data = list(file.iter())
        return data
    
def write_jsonlines(data, path):
    with jsonlines.open(path, "w") as writer:
        writer.write_all(data)

def read_json(path): 
    with open(path, 'r') as json_file: 
        data = json.load(json_file)   
        return data

def write_json(data, path):    
    with open(path, 'w') as writer:
        json.dump(data, writer, indent=4)

def read_file(path):
    with open(path, 'r') as file:
        lst = file.readlines()
        return lst

def write_list(lst, path):
    with open(path, 'w+') as f:
        for items in lst:
            f.write('%s\n' %items)

def compute_score_sklearn(gold_output, pred_output):
    binarizer.fit(gold_output)
    f1_score_macro = f1_score(binarizer.transform(gold_output),
                                      binarizer.transform(pred_output),
                                      average='macro')
    precision_macro = precision_score(binarizer.transform(gold_output),
                                      binarizer.transform(pred_output),
                                      average='macro')
    recall_macro = recall_score(binarizer.transform(gold_output),
                                      binarizer.transform(pred_output),
                                      average='macro')
    return precision_macro, recall_macro, f1_score_macro

def post_process_api_with_args(api_with_args_gold, api_with_args_pred):
    def align_lists(list1, list2):
        aligned_list1 = []
        aligned_list2 = []

        i, j = 0, 0

        while i < len(list1) or j < len(list2):
            if i < len(list1) and j < len(list2) and list1[i] == list2[j]:
                aligned_list1.append(list1[i])
                aligned_list2.append(list2[j])
                i += 1
                j += 1
            elif i < len(list1):
                aligned_list1.append(list1[i])
                aligned_list2.append("")
                i += 1
            else:
                aligned_list1.append("")
                aligned_list2.append(list2[j])
                j += 1

        return aligned_list1, aligned_list2
    
    api_names_gold = [api.split("(", 1)[0] for api in api_with_args_gold]
    api_names_pred = [api.split("(", 1)[0] for api in api_with_args_pred]
    
    if len(api_names_gold) == len(api_names_pred):
        return api_with_args_gold, api_with_args_pred
    
    try:
        api_names_gold, api_names_pred = align_lists(api_names_gold, api_names_pred)
        upd_api_with_args_gold, upd_api_with_args_pred = [], []
        for a_n in api_names_gold:
            if a_n == "":
                upd_api_with_args_gold.append("")
            else:
                upd_api_with_args_gold.append(api_with_args_gold.pop(0))
        for a_n in api_names_pred:
            if a_n == "":
                upd_api_with_args_pred.append("")
            else:
                upd_api_with_args_pred.append(api_with_args_pred.pop(0))
        return upd_api_with_args_gold, upd_api_with_args_pred
    except:
        return api_with_args_gold, api_with_args_pred
    