import json
import glob
import cv2
import PIL.Image
from tqdm import tqdm
import editdistance
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('ocr_folder', help='Path to folder containing OCR annotations')
parser.add_argument('documents_folder', help='Path to folder containing document images')
parser.add_argument('train_v1_json', help='Path to train_v1.0.json')
parser.add_argument('out_train_json')
parser.add_argument('out_val_json')

args = parser.parse_args()

def bbox_string(box, width, length):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / length)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / length))
    ]

def clean_text(text):
    replace_chars = ',.;:()-/$%&*' 
    for j in replace_chars:
        if text is not None:
            text = text.replace(j,'')
    return text

def harsh_find(answer_tokens, words):
    answer_raw = ''.join(answer_tokens)
    answer = ' '.join(answer_tokens)
    if len(answer_tokens)==1:
        for (ind,w) in enumerate(words):
            dist=0 if len(answer)<5 else 1
            if editdistance.eval(answer,w)<=dist:
                start_index=end_index=ind
                return start_index,end_index,w
    for (ind,w) in enumerate(words):
        if answer_raw.startswith(w): #Looks like words are split
            for inc in range(1,30):
                if ind+inc>=len(words):
                    break
                w=w+words[ind+inc]
                if len(answer_raw)>=5:
                        dist=1
                else:
                    dist=0
                start_index=ind
                end_index=ind+inc
                ext_list = words[start_index:end_index+1]
                extracted_answer = ' '.join(ext_list)

                if editdistance.eval(answer.replace(' ',''),extracted_answer.replace(' ',''))<=dist:
                    return start_index,end_index,extracted_answer
    return reverse_harsh_find(answer_tokens, words) 
    

def reverse_harsh_find(answer_tokens, words):
    answer_raw = ''.join(answer_tokens)
    answer = ''.join(answer_tokens)
    for (ind,w) in enumerate(words):
        if answer_raw.endswith(w): #Looks like words are split
            for inc in range(1,30):
                if ind-inc<0:
                    break
                w=words[ind-inc]+w
                if len(answer_raw)>=15:
                        dist=3
                elif len(answer_raw)>=5:
                        dist=1
                else:
                    dist=0
                start_index=ind-inc
                end_index=ind
                ext_list = words[start_index:end_index+1]
                extracted_answer = ' '.join(ext_list)

                if editdistance.eval(answer.replace(' ',''),extracted_answer.replace(' ',''))<=dist:
                    return start_index,end_index,extracted_answer
    return None,None,None

def get_answer_indices(ques_id,words, answer):
    count = 0
    answer_tokens = answer.split()
    end_index = None
    start_index = None
    words = [clean_text(x) for x in words]
    answer_tokens = [clean_text(x) for x in answer_tokens]
    answer = ' '.join(answer_tokens)
         
    if answer_tokens[0] in words:
        start_index = words.index(answer_tokens[0])
    if answer_tokens[-1] in words:
        end_index = words.index(answer_tokens[-1])
    if start_index is not None and end_index is not None:
        if start_index > end_index:
            if answer_tokens[-1] in words[start_index:]:    
                end_index = words[start_index:].index(answer_tokens[-1]) 
                end_index+=start_index
            else:
                #Last try
                start_index,end_index,extracted_answer = harsh_find(answer_tokens,words)
                return start_index,end_index,extracted_answer
     

        assert start_index<=end_index 
        extracted_answer = ' '.join(words[start_index:end_index+1])
        if answer.replace(' ','')!=extracted_answer.replace(' ',''):
            start_index,end_index,extracted_answer = harsh_find(answer_tokens,words)
            return start_index,end_index,extracted_answer
        else:
            return start_index, end_index, extracted_answer
        
        return None,None,None
    else:
        answer_raw = ''.join(answer_tokens)
        start_index,end_index,extracted_answer = harsh_find(answer_tokens,words)
        return start_index,end_index,extracted_answer
       
def find_candidate_lines(ocr_json,ans_json):
    pass

 
data = []
ocr_files = glob.glob(args.ocr_folder+"/*")
ocr_files = [x.split('.')[0] for x in ocr_files]
dict_img_qa = json.load(open(args.train_v1_json))
found = 0
nf = []
not_found = 0
img_id_covered = []

for datapt in tqdm(dict_img_qa["data"]):
    img_id = datapt["image"].split('/')[-1].split('.')[0]
    if img_id in img_id_covered:
        continue
    else:
        img_id_covered.append(img_id)
    img_qs = []
    questionId = []
    img_as = []

    for d in dict_img_qa["data"]:
        id_im = d["image"].split('/')[-1].split('.')[0]
        if id_im==img_id:
            img_qs.append(d["question"])
            questionId.append(d["questionId"])
            img_as.append(d["answers"][0])

    example = {}
    example["image_id"] = img_id
    example["qas"] = []
    words = []
    boxes = [] 
    boxes_norm = []
    line_indices = []
    lines_array = []

    ocr_file = glob.glob(args.ocr_folder+"/"+img_id+'.json')
    img_file = glob.glob(args.documents_folder+"/"+img_id+'.png')
    img = cv2.imread(img_file[0])
    length, width = img.shape[:2]
    ocr_json = json.load(open(ocr_file[0]))

    assert len(ocr_file)==1
    assert len(img_file)==1
    
    #Added boxes and context to the example 
    for obj in ocr_json['recognitionResults']:
        lines = obj['lines']
        idx = 0 
        for line in lines:
            lines_array.append(line['text'])
            for word in line['words']:
                words.append(word['text'].lower())
                line_indices.append(idx)        
                x1,y1,x2,y2,x3,y3,x4,y4 = word['boundingBox']
                new_x1 = min([x1,x2,x3,x4])
                new_x2 = max([x1,x2,x3,x4])
                new_y1 = min([y1,y2,y3,y4])
                new_y2 = max([y1,y2,y3,y4])
                boxes.append([new_x1,new_y1,new_x2,new_y2])
                box_norm = bbox_string([new_x1,new_y1,new_x2,new_y2], width, length)
                assert new_x2>=new_x1
                assert new_y2>=new_y1
                assert box_norm[2]>=box_norm[0]
                assert box_norm[3]>=box_norm[1]
                
                boxes_norm.append(box_norm)
                idx+=1
    example["context"] = words
    example["boxes"] = boxes_norm

    assert len(example["context"]) == len(example["boxes"])
    assert len(example["context"]) == len(line_indices)


    ques_counter = 1
    for qid in range(len(img_qs)):
        ques = img_qs[qid]
        ans = img_as[qid]
        ques_json = {}
        ques_json['qid'] = img_id+'-'+str(ques_counter)
        ques_counter+=1
        ques_json["question"] = ques.lower()
        ques_json["answer"] = []
        ans_json = {}
        ans_json["text"] = ans.lower()
        ques_json["answer"].append(ans_json)
        for ans_index in range(len(ques_json["answer"])):
            start_index, end_index, extracted_answer = get_answer_indices(ques_json['qid'],example["context"],ques_json["answer"][ans_index]["text"])
            replace_chars =',.;:()-/$%&*'
            ans=ans.lower()
            extracted_answer = clean_text(extracted_answer)
            ans = clean_text(ans)
            dist = editdistance.eval(extracted_answer.replace(' ',''),ans.replace(' ','')) if extracted_answer!=None else 1000
            if dist>5:
                start_index=None
            if start_index is not None:
                break
        if start_index is None or len(extracted_answer)>150 or extracted_answer=="":
            nf.append(img_id)
            not_found+=1
            start_index=None
            end_index=None
            continue
        else:
            found+=1
        ans_json["answer_start"] = start_index
        ans_json["answer_end"] = end_index
        example["qas"].append(ques_json)
    data.append(example) 

val_count=1
new_val = []
new_train = []


for i in tqdm(data):
    img_id = i['image_id']
    if val_count<=1000:
        new_val.append(i)
        val_count+=1
    else:
        new_train.append(i)
        
        
print("LEN VAL",len(new_val))
print("LEN TRAIN",len(new_train))

with open(args.out_train_json, "w") as fp:
    json.dump(new_train,fp)
with open(args.out_val_json, "w") as fp:
    json.dump(new_val,fp)

print("Answers found",found)
print("Answers not found",not_found)
