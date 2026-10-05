"""Vehicle identities for the CAN-Train-and-Test discovery pilot."""

# Published in can-train-and-test-v1.5/README.md.
CAN_TRAIN_TEST_VEHICLES: dict[str, tuple[str, str]] = {
    "set_01": ("chevrolet_impala", "chevrolet_silverado"),
    "set_02": ("chevrolet_traverse", "subaru_forester"),
    "set_03": ("chevrolet_silverado", "subaru_forester"),
    "set_04": ("subaru_forester", "chevrolet_traverse"),
}


def declared_vehicle_id(dataset: str, source_dir: str) -> str:
    pair = CAN_TRAIN_TEST_VEHICLES.get(dataset)
    if pair is None:
        return source_dir
    if source_dir.startswith("train_") or source_dir.startswith(("test_01_", "test_03_")):
        return pair[0]
    if source_dir.startswith(("test_02_", "test_04_")):
        return pair[1]
    return source_dir
