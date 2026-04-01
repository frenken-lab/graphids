from graphids import __main__ as cli_main


def test_lightning_command_is_routed() -> None:
    parser = cli_main._build_parser()
    ns, remaining = parser.parse_known_args(["fit", "--config", "x.yaml"])
    assert ns.kind == "lightning"
    assert ns.command_name == "fit"
    assert remaining == ["--config", "x.yaml"]


def test_module_command_is_routed() -> None:
    parser = cli_main._build_parser()
    ns, remaining = parser.parse_known_args(["analyze-from-spec", "--spec-file", "x.json"])
    assert ns.kind == "module"
    assert ns.module_name == "graphids.commands.analyze_from_spec"
    assert remaining == ["--spec-file", "x.json"]
