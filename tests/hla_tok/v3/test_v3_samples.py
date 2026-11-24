from PIL import Image

from hla_tok import BUDGETS
from scripts.hla_tok.v3 import samples


def test_progressive_contact_sheet_keeps_every_crop_at_original_pixel_scale(tmp_path):
    labels = ["source", *(f"K{budget}" for budget in BUDGETS)]
    for index, label in enumerate(labels):
        Image.new("RGB", (384, 216), (index, 30, 70)).save(tmp_path / f"{label}_crop.png")

    sheet = samples._progressive_sheet(tmp_path, labels)
    with Image.open(sheet) as image:
        assert image.width == 3 * 384
        panel_height = image.height // 3
        for index in range(len(labels)):
            column, row = index % 3, index // 3
            visible_crop = image.crop((column * 384, (row + 1) * panel_height - 216,
                                       (column + 1) * 384, (row + 1) * panel_height))
            with Image.open(tmp_path / f"{labels[index]}_crop.png") as original:
                assert visible_crop.tobytes() == original.tobytes()
