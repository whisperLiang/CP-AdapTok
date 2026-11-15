from cp_adaptok.runtime import build_model, preflight, read_config, seed_everything


def add_inference_options(parser, output):
    parser.add_argument("--config", default="configs/s1_tokenizer.yaml")
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--output", default=output)


def inference_config(args):
    config = read_config(args.config)
    if args.checkpoint:
        config["paths"]["checkpoint"] = args.checkpoint
    return config


def load_inference(config, args, purpose):
    device = args.device or config["device"]
    preflight(config, purpose, args.allow_untrained)
    seed_everything(config["seed"], device, config["gpu_memory_fraction"])
    model, status, _ = build_model(config, device, args.allow_untrained, args.checkpoint)
    return model.eval(), status
