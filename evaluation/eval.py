import json
import os
import datetime
import torch
from tqdm import tqdm
from vlmeval.smp import dump, tabulate, pd

@torch.no_grad()
def eval_dataset(model, dataset, dataset_name, model_name, verbose=False):

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_file = f"output/{model_name}_{dataset_name}_{timestamp}.xlsx"
    os.makedirs("output", exist_ok=True)
    res = {}
    lt = len(dataset.data)
    data_indices = [i for i in dataset.data["index"]]
    for i in tqdm(range(lt)):
        idx = dataset.data.iloc[i]["index"]
        if idx in res:
            continue

        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])

        response = model.generate(message=struct, dataset=dataset_name)
        if verbose:
            print(response, flush=True)
        res[idx] = response

    res = {k: res[k] for k in data_indices}

    data = dataset.data
    for x in data["index"]:
        assert x in res
    data["prediction"] = [str(res[x]) for x in data["index"]]
    if "image" in data:
        data.pop("image")

    dump(data, result_file)

    judge_kwargs = dict()
    eval_results = dataset.evaluate(result_file, **judge_kwargs)
    if eval_results is not None:
        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
        print(
            f"The evaluation of model {model_name} x dataset {dataset_name} has finished! "
        )
        print("Evaluation Results:")
    if isinstance(eval_results, dict):
        print("\n" + json.dumps(eval_results, indent=4))
    elif isinstance(eval_results, pd.DataFrame):
        if len(eval_results) < len(eval_results.columns):
            eval_results = eval_results.T
        print("\n" + tabulate(eval_results))