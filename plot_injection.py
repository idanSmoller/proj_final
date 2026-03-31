import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import Image


@dataclass
class ImageItem:
	path: Path
	axis: str
	power: Optional[float]
	power_text: str
	is_baseline: bool = False


def parse_power(token: str) -> Tuple[float, str]:
	"""
	Parse power token like "s0p5" into float 0.5 and display text "0.5".
	"""
	cleaned = token.strip()
	if cleaned.startswith("s"):
		cleaned = cleaned[1:]
	power_text = cleaned.replace("p", ".")
	return float(power_text), power_text


def parse_image_name(image_path: Path) -> ImageItem:
	stem = image_path.stem
	if "baseline" in stem:
		return ImageItem(
			path=image_path,
			axis="baseline",
			power=None,
			power_text="0.0",
			is_baseline=True,
		)

	parts = stem.split("_")
	if "axis" in parts:
		parts.remove("axis")
	axis = parts[0]
	power_value = None
	power_text = ""

	if len(parts) > 1:
		power_value, power_text = parse_power(parts[1])

	return ImageItem(
		path=image_path,
		axis=axis,
		power=power_value,
		power_text=power_text,
		is_baseline=False,
	)


def build_two_extremes_sequence(
	by_axis: Dict[str, List[ImageItem]], baseline: Optional[ImageItem],
	left_indices: Optional[List[int]] = None, right_indices: Optional[List[int]] = None
) -> Tuple[List[ImageItem], str]:
	"""
	Build one progression row:
	left extreme (descending strength) -> baseline -> right extreme (ascending strength)
	left_indices/right_indices: 1-based indices to select from each side (if None, use all)
	"""
	axis_names = sorted(by_axis.keys())
	left_axis, right_axis = axis_names[0], axis_names[1]

	left = sorted(
		by_axis[left_axis],
		key=lambda x: (x.power is None, x.power),
		reverse=True,
	)
	right = sorted(
		by_axis[right_axis],
		key=lambda x: (x.power is None, x.power),
	)

	# Select only specified indices (convert 1-based to 0-based)
	if left_indices is not None:
		left = [left[i-1] for i in left_indices if 0 < i <= len(left)]
	if right_indices is not None:
		right = [right[i-1] for i in right_indices if 0 < i <= len(right)]

	sequence = left.copy()
	if baseline is not None:
		sequence.append(baseline)
	sequence.extend(right)

	subtitle = f"Left extreme: {left_axis} | Right extreme: {right_axis}"
	return sequence, subtitle


def build_generic_sequence(
	by_axis: Dict[str, List[ImageItem]], baseline: Optional[ImageItem]
) -> Tuple[List[ImageItem], str]:
	sequence: List[ImageItem] = []
	if baseline is not None:
		sequence.append(baseline)

	for axis_name in sorted(by_axis.keys()):
		axis_items = sorted(
			by_axis[axis_name],
			key=lambda x: (x.power is None, x.power),
		)
		sequence.extend(axis_items)

	subtitle = "Axis labels are parsed from split('_')[0]"
	return sequence, subtitle


def panel_label(item: ImageItem, is_first: bool, is_last: bool) -> str:
	if item.is_baseline:
		return "baseline"
	parts = [f"{item.axis}", f"s={item.power_text}"]
	if is_first:
		parts.append("(axis extreme)")
	if is_last:
		parts.append("(axis extreme)")
	return "\n".join(parts)


def plot_prompt_progression(prompt_dir: Path, output_dir: Path,
							left_indices: Optional[List[int]] = None,
							right_indices: Optional[List[int]] = None) -> Optional[Path]:
	image_paths = sorted(prompt_dir.glob("*.png"))
	if not image_paths:
		return None

	parsed = [parse_image_name(path) for path in image_paths]
	baseline = next((item for item in parsed if item.is_baseline), None)

	by_axis: Dict[str, List[ImageItem]] = {}
	for item in parsed:
		if item.is_baseline:
			continue
		by_axis.setdefault(item.axis, []).append(item)

	if not by_axis and baseline is None:
		return None

	if len(by_axis) == 2:
		sequence, subtitle = build_two_extremes_sequence(by_axis, baseline, left_indices, right_indices)
	else:
		sequence, subtitle = build_generic_sequence(by_axis, baseline)

	if not sequence:
		return None

	cols = len(sequence)
	fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.8), squeeze=False)

	for idx, item in enumerate(sequence):
		ax = axes[0][idx]
		with Image.open(item.path) as img:
			ax.imshow(img)
		ax.axis("off")

		label = panel_label(item, is_first=(idx == 0), is_last=(idx == cols - 1))
		ax.set_title(label, fontsize=10)

	fig.suptitle(f"{prompt_dir.name}\n{subtitle}", fontsize=12)
	fig.tight_layout()

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{prompt_dir.name}_progression.png"
	fig.savefig(out_path, dpi=200)
	plt.close(fig)
	return out_path



def parse_indices(indices_str: Optional[str]) -> Optional[List[int]]:
	if not indices_str:
		return None
	# Accept comma-separated or range (e.g., 1,3,5 or 1-3)
	indices = []
	for part in indices_str.split(","):
		part = part.strip()
		if "-" in part:
			start, end = part.split("-")
			indices.extend(list(range(int(start), int(end)+1)))
		elif part:
			indices.append(int(part))
	return indices if indices else None

def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot side-by-side injection progression with axis/power labels parsed from filenames."
	)
	parser.add_argument(
		"--outcomes-dir",
		type=Path,
		default=Path("outputs/injection_results/outcomes"),
		help="Folder containing per-prompt output folders.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("outputs/injection_results/progression_plots"),
		help="Where to save progression plots.",
	)
	parser.add_argument(
		"--indices",
		type=str,
		default=None,
		help="Comma-separated or range (e.g., 1,3,5 or 1-3)",
	)

	args = parser.parse_args()

	total_images = sum(len(list(p.glob("*.png"))) for p in args.outcomes_dir.iterdir() if p.is_dir())
	axis_imgs = (total_images-1)//2 + 1
	right_indices = parse_indices(args.indices)
	if right_indices is not None:
		left_indices = [axis_imgs - i for i in right_indices]  # Mirror for left side (assuming 5 total)
	else:
		left_indices = None

	prompt_dirs = [p for p in sorted(args.outcomes_dir.iterdir()) if p.is_dir()]
	if not prompt_dirs:
		print(f"No prompt folders found in {args.outcomes_dir}")
		return

	created = 0
	for prompt_dir in prompt_dirs:
		out = plot_prompt_progression(prompt_dir, args.output_dir, left_indices, right_indices)
		if out is not None:
			created += 1
			print(f"Saved: {out}")

	print(f"Done. Created {created} progression plot(s).")


if __name__ == "__main__":
	main()
