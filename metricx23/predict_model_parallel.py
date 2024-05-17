import argparse
import json
from metricx23 import models
import torch
import transformers
from datetime import datetime
from dataclasses import dataclass

def construct_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--max_input_length", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--input_files", type=str)
    parser.add_argument("--output_files", type=str)
    parser.add_argument("--qe", action='store_true')
    parser.add_argument('--output_follow_input_file_order', default=True, action='store_true')
    parser.add_argument('--output_not_follow_input_file_order', dest='output_follow_input_file_order', action='store_false')

    args.input_files = args.input_files.split(',')
    args.input_files = args.output_files.split(',')
    args = parser.parse_args()
    return args

class JsonlUntokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, is_qe, max_input_length):
        with open(path, "r") as f:
            self.data = f.readlines()
        self.data = [json.loads(example) for example in self.data]
        if is_qe:
            self.data = [{**{"input": f'candidate: {example["hypothesis"]} source: {example["source"]}', "input_file_order": i}, **example} for i, example in enumerate(self.data)]
        else:
            self.data = [{**{"input": f'candidate: {example["hypothesis"]} reference: {example["reference"]}', "input_file_order": i}, **example} for i, example in enumerate(self.data)]

        data_tokens = tokenizer([example["input"] for example in self.data], max_length=max_input_length, truncation=True, padding=False)["input_ids"]
        # remove eos token
        self.data = [{**example, **{"input_ids": torch.tensor(input_ids[:-1])}} for example, input_ids in zip(self.data, data_tokens)]
        del data_tokens
        # sort by input_ids length, for faster inference
        self.data = sorted(self.data, key=lambda x: len(x["input_ids"]))

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
    model = models.MT5ForRegression.from_pretrained(args.model_name_or_path, device_map='balanced')
    model.eval()
    print(f"{datetime.now().strftime('%H:%M:%S')} loaded model")

    for input_file, output_file in zip(args.input_files, args.output_files):
        data_collator = DefaultDataCollator(tokenizer=tokenizer)
        dataset = JsonlUntokenizedDataset(args.input_file, tokenizer, args.qe, args.max_input_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator, drop_last=False)
        print(f"{datetime.now().strftime('%H:%M:%S')} loaded {len(data_loader)} batches, total {len(dataset)} samples")

        predictions = []
        for batch in  data_loader:
            batch_predictions = model(**batch).predictions.detach().cpu().tolist()
            predictions.extend(batch_predictions)
        print(f"{datetime.now().strftime('%H:%M:%S')} completed {len(data_loader)} batches, total {len(predictions)} predictions")

        outputs = [{**{k:v for k,v in example.items()},**{'prediction': pred}} for example, pred in zip(dataset.data, predictions)]
        if args.output_follow_input_file_order:
            outputs = sorted(outputs, key=lambda x: x["input_file_order"])
        with open(args.output_file, "w") as f:
            for o in outputs:
                json.dump({k:v for k,v in o.items() if k not in ['input','input_ids', "input_file_order"]}, f)
                f.write("\n")





