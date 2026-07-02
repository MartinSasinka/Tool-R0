import json, re, ast_utils

def get_deli_sep_str_list(text, deli = ','):
    def find(s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    comma_indexes = find(text, deli)
    valid_comma_indexes = []

    for idx in comma_indexes:
        valid_flag = True
        lfs, rfs = text[:idx], text[idx + 1:]

        # Delimeter not inside quotes
        quotes_count_lfs = lfs.count('"')
        if not quotes_count_lfs % 2 == 0:
            valid_flag = False

        if valid_flag:
            valid_comma_indexes.append(idx)
    parts = []
    temp_idx = 0
    for idx in valid_comma_indexes:
        parts.append(text[temp_idx:idx])
        temp_idx = idx + 2
    parts.append(text[temp_idx:])
    parts = [p.strip() for p in parts]
    return parts

def process_slot_value(slot_value):
    if slot_value.startswith("\'"):
        slot_value = slot_value[1:]
    if slot_value.endswith("\'"):
        slot_value = slot_value[:-1]
    if not slot_value.startswith("\""):
        slot_value =  "\"" + slot_value
    if not slot_value.endswith("\""):
        slot_value = slot_value + "\""
    sub_slot_value = slot_value[1:-1].replace("\"","'")
    return sub_slot_value

def ground_seq_nested_repsonse(api_list):
    def check_label_in_slot(label, slot_v):
        if slot_v.startswith("$var"):
            if '.' in slot_v:
                lbl_slot = slot_v.split(".", 1)[0].replace("$", "")
                if lbl_slot == label:
                    return True
        return False 

    label_api_map = {}
    for api in api_list:
        if api['name'] == 'varResult':
            continue
        if api['name'] == 'var_result':
            continue
        if 'label' in api:
            lbl = api['label'].replace("$", "")
            label_api_map[lbl] = api['name']

    grounded_api_list = []

    for api in api_list:
        if api['name'] == 'var_result':
            continue
        temp_arguments = {}
        if 'arguments' in api:
            arg_dict = api['arguments']
        elif 'parameters' in api:
            arg_dict = api['parameters']
        else:
            arg_dict = {}
        for s_n, s_v in arg_dict.items():
            for l, a in label_api_map.items():
                if type(s_v) == str and check_label_in_slot(l, s_v):
                    s_v = s_v.replace(l, a)
                elif type(s_v) == list:
                    new_s_v = []
                    for v in s_v:
                        if type(v) == str and check_label_in_slot(l, v):
                            v = v.replace(l, a)
                        elif type(v) == dict and check_label_in_slot(l, json.dumps(v)):
                            v = json.loads(json.dumps(v).replace(l, a))
                        new_s_v.append(v)
                    s_v = new_s_v
            temp_arguments[s_n] = s_v

        grounded_api_list.append({
            'name': api['name'],
            'arguments': temp_arguments
        })
    return grounded_api_list

def parse_granite_20b_function_calling_output(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item['output'])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred = item['generated_text'].strip().replace('ASSISTANT', '').strip()
        pred_str_list = pred.split('<function_call>')
        pred_dict_list = [json.loads(p) for p in pred_str_list if p]
        pred_dict_list = [p for p in pred_dict_list if not p['name'] == "var_result"]
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list) if 'label' in pred else pred_dict_list
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list, num_errors_parsing_pred_intent, pred_has_parsing_errors 

def parse_granite_3_output(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item['output'])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred_dict_list = json.loads(item['generated_text'])
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list) if 'label' in item['generated_text'] else pred_dict_list
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True
    return pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list, num_errors_parsing_pred_intent, pred_has_parsing_errors 

def parse_granite_3_1_output(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item['output'])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        generated_text = item['generated_text'].replace("<|tool_call|>", "").replace("<tool_call>", "")
        pred_dict_list = json.loads(generated_text)
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list) if 'label' in generated_text else pred_dict_list
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True
    return pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list, num_errors_parsing_pred_intent, pred_has_parsing_errors 

def parse_llama_3_output(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item['output'])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        gen_text = item['generated_text'].strip()
        if gen_text.endswith("'"):
            gen_text = gen_text[:-1]
        if not gen_text.startswith('['):
            gen_text = '[' + gen_text
        if not gen_text.endswith(']'):
            gen_text = gen_text + ']' 
        
        pred_dict_list = json.loads(gen_text)
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list) if 'label' in item['generated_text'] else pred_dict_list
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list, num_errors_parsing_pred_intent, pred_has_parsing_errors 

def parse_llama_3_70b_instruct(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item['output'])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]
    ## Pred
    try:
        pred = item['generated_text'].strip()
        pred_dict_list = json.loads(pred)        
        pred_dict_list = [p for p in pred_dict_list if not p['name'] == "var_result"]
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list) if 'label' in pred else pred_dict_list
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        try:
            pred = item['generated_text'].strip()
            if pred.startswith("[") and pred.endswith("]"):
                pred = pred[1:-1]
                pred_list = pred.split("),")
                new_pred_list = []
                for p in pred_list:
                    if p.strip().endswith(')'):
                        new_pred_list.append(p)
                    else:
                        new_pred_list.append(p + ")")
                pred_func_calls = []
                for p in new_pred_list:
                    intent = p.split("(", 1)[0]
                    slot_str = p.split("(", 1)[1][:-1]
                    slots = get_deli_sep_str_list(slot_str)
                    arg_dict = {}
                    for s in slots:
                        s_n, s_v = s.split("=")[0].strip(), s.split("=")[1].strip()
                        arg_dict[s_n] = process_slot_value(s_v)
                    pred_func_calls.append(json.dumps({
                        "name": intent,
                        "arguments": arg_dict
                    }))
            else:
                num_errors_parsing_pred_intent += 1
                pred_has_parsing_errors = True
        except:
            num_errors_parsing_pred_intent += 1
            pred_has_parsing_errors = True

    return pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list, num_errors_parsing_pred_intent, pred_has_parsing_errors 

def parse_mistral_7b_instruct_v0_3(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []

    ## Gold
    gold_dict_list = json.loads(item['output'])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred = item['generated_text'].strip()
        pred_dict_list = json.loads(pred)
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list)
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        try:
            pred = item['generated_text'].strip()
            pred_dict_list = json.loads(pred.replace("\n", "").replace("\_", "_"))
            if skip_grounding:
                pred_func_calls = [json.dumps(func) for func in pred_dict_list]
            else:
                pred_func_calls = ground_seq_nested_repsonse(pred_dict_list)
                pred_func_calls = [json.dumps(func) for func in pred_func_calls]
        except:
            num_errors_parsing_pred_intent += 1
            pred_has_parsing_errors = True

    return pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list, num_errors_parsing_pred_intent, pred_has_parsing_errors 

def parse_hermes_2_pro_mistral_7B(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item['output'])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]
    
    ## Pred
    try:
        pred = item['generated_text'].strip()
        if pred.startswith("tool_call\n{"):
            pred = pred.replace("tool_call\n{", "<tool_call>\n{")
        func_str_list = re.findall(r'<tool_call>(.*?)</tool_call>', pred, re.DOTALL)
        
        pred_dict_list = []
        for p in func_str_list:
            try:
                pred_dict_list.append(json.loads(p.strip()))
            except:
                continue

        assert len(pred_dict_list) > 0, "parsing issue"
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list)
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True
    return pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list, num_errors_parsing_pred_intent, pred_has_parsing_errors 

def parse_xLAM_1b_fc_r(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item['output'])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred = item['generated_text'].strip()
        pred = pred.replace("[BEGIN OF ANSWER]","").replace("[BEGIN OF FUNCTION CALLS]", "").replace("[BEGIN OF FUNCTION CALL]", "").replace("[BEGIN OF ACTIONS]", "").replace("[BEGIN OF ACTION]", "").strip()
        tool_dict = json.loads(pred)
        assert "tool_calls" in tool_dict, "parsing error"
        pred_dict_list = tool_dict['tool_calls']
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list)
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True
    return pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list, num_errors_parsing_pred_intent, pred_has_parsing_errors 


def parse_Hammer2_0_7b(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []

    ## Gold
    gold_dict_list = json.loads(item['output'])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred = item["generated_text"].replace("```", "").strip()
        pred_dict_list = json.loads(pred)
        assert len(pred_dict_list) > 0, "parsing issue"

        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list)
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
            
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list, num_errors_parsing_pred_intent, pred_has_parsing_errors 


def parse_ToolAce(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []

    ## Gold
    gold_dict_list = json.loads(item['output'])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred = item["generated_text"]
        pred_dict_list = ast_utils.ast_parse(pred)
        assert len(pred_dict_list) > 0, "parsing issue"

        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list)
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
            
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list, num_errors_parsing_pred_intent, pred_has_parsing_errors 


def parse_deepseek_output(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item['output'])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred_dict_list = json.loads(item['generated_text'])
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list) if 'label' in item['generated_text'] else pred_dict_list
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True
    return pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list, num_errors_parsing_pred_intent, pred_has_parsing_errors 