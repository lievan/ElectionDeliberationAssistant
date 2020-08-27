import gensim
import nltk
from nltk.corpus import stopwords

from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig
from transformers import RobertaTokenizer
import torch
from torch import nn
import numpy as np

nltk.download('stopwords')

stopword = set(stopwords.words('english'))


class SemanticSearcher:
    def __init__(self, path):

        self.embeddings_dict = {}
        print('Loading word vectors...')
        with open(path, 'r') as f:
            for line in f:
                values = line.split()
                split = len(values) - 300
                word = values[split - 1]
                vector = np.asarray(values[split:], "float32")
                self.embeddings_dict[word] = vector
        print('Word vectors loaded')

    def get_mean_vector(self, text):

        words = self.preprocess_text(text)

        word_list = []
        for word in words:
            if word in self.embeddings_dict:
                word_list.append(word)

        word_vectors = []
        for word in word_list:
            word_vectors.append(self.embeddings_dict[word])
        return np.sum(word_vectors, axis=0)

    def predict_distance(self, post1, post2):

        vector0 = self.get_mean_vector(post1)
        vector1 = self.get_mean_vector(post2)
        norm_vector0 = vector0
        norm_vector1 = vector1
        similarity = 1 - spatial.distance.cosine(norm_vector0, norm_vector1)
        return similarity

    def preprocess_text(self, text):
        # helper method to preprocess text
        text = gensim.utils.simple_preprocess(text)
        return_text = []
        for tok in text:
            if not tok in stopword:
                return_text.append(tok)
        return return_text

    def get_recs(self, new_text, text_rec_list, top_n=200):

        output_examples = []

        for post1_text in text_rec_list:
            output_examples.append([self.predict_distance(post1_text, new_text), post1_text])

        output_examples = sorted(output_examples, key=lambda x: x[0], reverse=True)

        outputs = []
        for example in output_examples[:top_n]:
            outputs.append(example[1])
        return outputs


class argBERT(nn.Module):

    def __init__(self, model_name, device: str = None):

        super(argBERT, self).__init__()
        self.argBERT = RobertaForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name, bos_token='<s>', eos_token='</s>',
                                                          unk_token='<unk>',
                                                          pad_token='<pad>', mask_token='mask_token', sep_token="</s>",
                                                          cls_token='<s>')
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.argBERT.to(self.device)
        self.best_accuracy_score = 1000

    def get_response_recs(self, text, rec_list, top_n=10):

        self.argBERT.eval()

        input_ids = []
        attention_masks = []

        output_examples = []

        for response in rec_list:
            output_examples.append([response])

            # encode parent_text and child_text representation with tokenizer
            encoded_input = self.tokenizer.encode_plus(
                response,
                text,
                add_special_tokens=True,
                max_length=128,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation="longest_first"
            )

            # move input IDs to GPU
            input_ids.append(torch.tensor(encoded_input['input_ids']).to(self.device))
            attention_masks.append(torch.tensor(encoded_input['attention_mask']).to(self.device))

        predicted_logits = []

        # make our predictions for every post in the map
        with torch.no_grad():
            for input_id, attention_mask in zip(input_ids, attention_masks):
                output = self.argBERT(input_id,
                                      token_type_ids=None,
                                      attention_mask=attention_mask)
                logit = output[0].detach().cpu().numpy()
                predicted_logits.append(logit)

        for i in range(len(output_examples)):
            output_examples[i].append(predicted_logits[i])

        # sort by taxonomic distance
        output_examples = sorted(output_examples, key=lambda x: x[1])

        return output_examples[:top_n]


def get_arg_list(trump_file_path, biden_file_path):
    with open(trump_file_path, 'r') as f:
        trump_list = [line.rstrip('\n') for line in f]

    with open(biden_file_path, 'r') as f:
        biden_list = [line.rstrip('\n') for line in f]

    return trump_list, biden_list


def get_responses(new_text, trump_list, biden_list, argBERT_model, similarity_model, top_n=5):
    topical_trump = similarity_model.get_recs(new_text, trump_list)

    topical_biden = similarity_model.get_recs(new_text, biden_list)

    trump_thinks = argBERT_model.get_response_recs(new_text, topical_trump, top_n=top_n)

    biden_thinks = argBERT_model.get_response_recs(new_text, topical_biden, top_n=top_n)

    for i in range(len(trump_thinks[:top_n])):
        trump_thinks[i][0] = trump_thinks[i][0].replace("PRO ", "")
        trump_thinks[i][0] = trump_thinks[i][0].replace("CON ", "")
        biden_thinks[i][0] = biden_thinks[i][0].replace("PRO ", "")
        biden_thinks[i][0] = biden_thinks[i][0].replace("CON ", "")

    return trump_thinks, biden_thinks