import argparse
import json
from metricx24 import models
import torch
import transformers
from datetime import datetime
from dataclasses import dataclass
import torch
import time

def construct_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--max_input_length", type=int)
    parser.add_argument("--input_lengths", type=str)
    parser.add_argument("--batch_sizes", type=str)
    parser.add_argument("--input_files", type=str)
    parser.add_argument("--output_files", type=str)
    parser.add_argument("--qe", action='store_true')
    parser.add_argument('--output_follow_input_file_order', default=True, action='store_true')
    parser.add_argument('--output_not_follow_input_file_order', dest='output_follow_input_file_order', action='store_false')

    args = parser.parse_args()
    args.input_files = args.input_files.split(',')
    args.output_files = args.output_files.split(',')
    args.input_lengths = [int(l) for l in args.input_lengths.split(',')]
    args.batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
    return args


def create_dataloaders(path, tokenizer, is_qe, max_input_length, input_lengths, batch_sizes):
    data = open(path, "r").readlines()
    data = [json.loads(example) for example in data]

    if is_qe:
        data = [{**{"input": f'source: {example["source"]} candidate: {example["hypothesis"]}', "input_file_order": i}, **example} for i, example in enumerate(data)]
    else:
        data = [{**{"input": f'source: {example["source"]} candidate: {example["hypothesis"]} reference: {example["reference"]}', "input_file_order": i}, **example} for i, example in enumerate(data)]

    data_tokens = tokenizer([example["input"] for example in data], max_length=max_input_length, truncation=True, padding=False)["input_ids"]
    # remove eos token
    data = [{**example, **{"input_ids": torch.tensor(input_ids[:-1])}} for example, input_ids in zip(data, data_tokens)]
    del data_tokens
    # sort by input_ids length, for faster inference
    data = sorted(data, key=lambda x: len(x["input_ids"]))
    dataloaders = []
    current_loader_ind=0
    current_loader_data=[]

    for example in data:
        if len(example["input_ids"])>args.input_lengths[current_loader_ind] and len(current_loader_data)>0:
            dataloaders.append(torch.utils.data.DataLoader(DefaultDataset(current_loader_data), batch_size=args.batch_sizes[current_loader_ind], collate_fn=DefaultDataCollator(tokenizer=tokenizer), drop_last=False))
        while len(example["input_ids"])>args.input_lengths[current_loader_ind]:
            current_loader_ind=current_loader_ind+1
            current_loader_data=[]
        current_loader_data.append(example)

    if len(current_loader_data)>0:
        dataloaders.append(torch.utils.data.DataLoader(DefaultDataset(current_loader_data), batch_size=args.batch_sizes[current_loader_ind], collate_fn=DefaultDataCollator(tokenizer=tokenizer), drop_last=False))
    return data, dataloaders




class DefaultDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind]

@dataclass
class DefaultDataCollator():
    tokenizer: transformers.AutoTokenizer
    def __call__(self, instances):
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),)



if __name__ == "__main__":
    args = construct_arguments()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    model = models.MT5ForRegression.from_pretrained(args.model_name_or_path, device_map='balanced', torch_dtype="auto")
    model.eval()
    print(f"{datetime.now().strftime('%H:%M:%S')} loaded model")

    for input_file, output_file in zip(args.input_files, args.output_files):
        sorted_data, data_loaders = create_dataloaders(input_file, tokenizer, args.qe, args.max_input_length, args.input_lengths, args.batch_sizes)
        print(f"{datetime.now().strftime('%H:%M:%S')} created {len(data_loaders)} data_loaders with sizes: {[len(data_loader) for data_loader in data_loaders]}")

        predictions = []
        for data_loader in data_loaders:
            for batch in data_loader:
                start_time = time.time()
                batch_predictions = model(**batch).predictions.detach().cpu().tolist()
                predictions.extend(batch_predictions)
                print(f"{datetime.now().strftime('%H:%M:%S')} completed one batch of shape {batch['input_ids'].shape}, gpu utilizations: {[torch.cuda.utilization(device=d) for d in range(torch.cuda.device_count())]}, takes {time.time()-start_time}s")
            print(f"{datetime.now().strftime('%H:%M:%S')} completed {len(data_loader)} batches, total {len(predictions)} predictions")

        outputs = [{**{k:v for k,v in example.items()},**{'prediction': pred}} for example, pred in zip(sorted_data, predictions)]
        if args.output_follow_input_file_order:
            outputs = sorted(outputs, key=lambda x: x["input_file_order"])
        with open(output_file, "w") as f:
            for o in outputs:
                json.dump({k:v for k,v in o.items() if k not in ['input','input_ids', "input_file_order"]}, f)
                f.write("\n")





