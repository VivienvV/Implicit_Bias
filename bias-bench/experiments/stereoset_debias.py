import argparse
import json
import os

import torch
import transformers

from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative, _is_self_debias


thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs StereoSet benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="SentenceDebiasForMaskedLM",
    choices=[
        "SentenceDebiasBertForMaskedLM",
        "SentenceDebiasAlbertForMaskedLM",
        "SentenceDebiasRobertaForMaskedLM",
        "SentenceDebiasGPT2LMHeadModel",
        "INLPBertForMaskedLM",
        "INLPAlbertForMaskedLM",
        "INLPRobertaForMaskedLM",
        "INLPGPT2LMHeadModel",
        "CDABertForMaskedLM",
        "CDAAlbertForMaskedLM",
        "CDARobertaForMaskedLM",
        "CDAGPT2LMHeadModel",
        "DropoutBertForMaskedLM",
        "DropoutAlbertForMaskedLM",
        "DropoutRobertaForMaskedLM",
        "DropoutGPT2LMHeadModel",
        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",
    ],
    help="Model to evalute (e.g., SentenceDebiasBertForMaskedLM).",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    choices=["bert-base-uncased", "albert-base-v2", "roberta-base", "gpt2"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--bias_direction",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed bias direction for SentenceDebias.",
)
parser.add_argument(
    "--projection_matrix",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed projection matrix for INLP.",
)
parser.add_argument(
    "--load_path",
    action="store",
    type=str,
    help="Path to saved CDA or Dropout model checkpoint.",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=1,
    help="The batch size to use during StereoSet intrasentence evaluation.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    type=str,
    default='gender',
    choices=["gender", "religion", "race"],
    help="The type of bias to mitigate.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=None,
    help="RNG seed. Used for logging in experiment ID.",
)

parser.add_argument(
    "--input_file",
    action="store",
    type=str,
    default="../../project_code/gender_data.json",
    help="The json file that contains the data to evaluate"
)

parser.add_argument(
    "--experiment_name",
    action="store",
    type=str,
    default="implicit_gender_debias",
    help="The name of the experiment- which will be added to the experiment id"
)

parser.add_argument(
    "--debiasing_prefixes",
    action="store",
    type=str,
    default={"gender": "The following text discriminates against people because of their gender: "},
    help="the prefixed to use for debiasing"
)

parser.add_argument(
    "--bias_position",
    action="store",
    type=str,
    default=13,
    help="the number of tokens in the prompt"
)

if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="stereoset",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
        seed=args.seed,
    )

    print("Running StereoSet:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_direction: {args.bias_direction}")
    print(f" - projection_matrix: {args.projection_matrix}")
    print(f" - load_path: {args.load_path}")
    print(f" - batch_size: {args.batch_size}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - seed: {args.seed}")
    print(f" - experiment_id : {experiment_id}")
    print(f" - experiment_name : {args.experiment_name}")
    print(f" - debiasing prefixes : {args.debiasing_prefixes}")

    kwargs = {}
    if args.bias_direction is not None:
        # Load the pre-computed bias direction for SentenceDebias.
        bias_direction = torch.load(args.bias_direction)
        kwargs["bias_direction"] = bias_direction

    if args.projection_matrix is not None:
        # Load the pre-computed projection matrix for INLP.
        projection_matrix = torch.load(args.projection_matrix)
        kwargs["projection_matrix"] = projection_matrix

    model = getattr(models, args.model)(
        args.load_path or args.model_name_or_path, **kwargs
    )

    if _is_self_debias(args.model):
        model._model.eval()
    else:
        model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Use self-debiasing name.
    bias_type = args.bias_type
    if bias_type == "race":
        bias_type = "race-color"

    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=args.input_file,#"{args.persistent_dir}/data/stereoset/test.json",,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        is_generative=_is_generative(args.model),
        is_self_debias=_is_self_debias(args.model),
        bias_type=bias_type,
        debiasing_prefixes = json.loads(args.debiasing_prefixes),
        bias_position = int(args.bias_position)
    )
    results = runner()

    os.makedirs(f"{args.persistent_dir}/results/stereoset", exist_ok=True)
    with open(
        f"{args.persistent_dir}/results/stereoset/{experiment_id}_{args.experiment_name}.json", "w"
    ) as f:
        json.dump(results, f, indent=2)
