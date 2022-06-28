import torch
from fairseq.data.data_utils import collate_tokens
import numpy as np
import re

NLI_MODEL = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
NLI_MODEL.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1)
NLI_MODEL.to(device)


'''
    Returns ([[contradiction, neutral, entailment]], argmax)
'''
def nli_metric_batch(batch_of_pairs):
    # batch_of_pairs = [
    #     ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
    #     ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
    #     ['potatoes are awesome.', 'I like to run.'],
    #     ['Mars is very far from earth.', 'Mars is very close.'],
    # ]

    encoded_tokens = [NLI_MODEL.encode(pair[0], pair[1]) for pair in batch_of_pairs]
    encoded_tokens = [tokens[:min(len(tokens), 512)] for tokens in encoded_tokens] # trucate any long seq
    batch = collate_tokens(
        encoded_tokens, pad_idx=1
    )

    logprobs = NLI_MODEL.predict('mnli', batch)
    logits = softmax(logprobs)
    labels = logits.argmax(dim=1) # logprobs.argmax(dim=1)

    return logits.tolist(), labels.tolist()

    

def nli_metric(premise, hypothesis):

    # Encode a pair of sentences and make a prediction
    # tokens = NLI_MODEL.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
    tokens = NLI_MODEL.encode(premise, hypothesis)

    seq_len = min(len(tokens), 512)

    logits = NLI_MODEL.predict('mnli', tokens[:seq_len])
    logits = softmax(logits)
    label = logits.argmax()  # 0: contradiction

    return logits.tolist(), label.tolist()


# ('As much as', 'CARDINAL')
# ('About 20', 'CARDINAL')
# ('67', 'CARDINAL'), 
# ('14,000 meters', 'QUANTITY')    vs     ('1.4 kilometers', 'QUANTITY')

def ner_metric(named_entities, prompt_wiki_candidates):
    
    wiki_text = " ".join(prompt_wiki_candidates).lower()

    # TODO improve the NE match here
    # hanlde DATE, TIME, etc better! appears a lot but handled poorly

    existing_correct_ne = []
    for ent in named_entities:
        ent_text = ent[0].lower()
        if 'the ' in ent_text:
            ent_text = ent_text.replace('the ', "")

        if ent_text in wiki_text:
            existing_correct_ne.append(ent)
        elif any([bool(word in wiki_text) for word in ent_text.split(" ") if ent[1] == 'PERSON']):
            # handle shorter forms of same NE: Exists "Marcus Morgan Bentley", but NE is "Marcus Bentley" or "Bentley"
            existing_correct_ne.append(ent)
        elif ent[1] == 'DATE':
            date_str = re.sub(r"[,.;@#?!&$]+\ *", " ", ent_text)
            date_str = date_str.replace("st", "")
            date_str = date_str.replace("nd", "")
            date_str = date_str.replace("th", "")
            date_str = date_str.replace("of", "")
            date_tokens = date_str.split(" ")

            if all([bool(token in wiki_text) for token in date_tokens]):
                existing_correct_ne.append(ent)
        


    correct_ratio = len(existing_correct_ne)/ len(named_entities)

    return correct_ratio

    
def ie_metric(claims, evidences):
    return NotImplementedError



if __name__ == '__main__':

    print("Hi")