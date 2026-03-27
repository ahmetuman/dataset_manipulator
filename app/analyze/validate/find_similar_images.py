import os
import sys
import hashlib
from PIL import Image
import imagehash


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PHASH_THRESHOLD = 5

class SimilarityCalculator:
    def __init__(self, root_directory, test_run, log_file = "similars.txt"):
        self.root_directory = root_directory
        self.test_run = test_run
        self.log_file = log_file

        with open(self.log_file, "w") as f:
            f.write("")

    def _log_group(self, text):
        with open(self.log_file, "a") as f:
            f.write(text + "\n")

    def _collect_images(self, root_directory):
        images = []
        for dirpath, _, filenames in os.walk(root_directory):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS:
                    images.append(os.path.join(dirpath, filename))
        return sorted(images)


    def _compute_file_hash(self, filepath):
        hasher = hashlib.md5()
        with open(filepath, "rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


    def _compute_perceptual_hash(self, filepath):
        image = Image.open(filepath)
        return imagehash.phash(image)


    def _get_image_info(self, filepath):
        with Image.open(filepath) as image:
            width, height = image.size
        size_bytes = os.path.getsize(filepath)
        return width, height, size_bytes


    def _find_label_path(self, image_path):
        parent = os.path.dirname(image_path)
        grandparent = os.path.dirname(parent)
        stem = os.path.splitext(os.path.basename(image_path))[0]

        if os.path.basename(parent) == "images":
            label_path = os.path.join(grandparent, "labels", stem + ".txt")
            if os.path.exists(label_path):
                return label_path

        label_path = os.path.join(parent, stem + ".txt")
        if os.path.exists(label_path):
            return label_path

        return None


    def _pick_keeper_and_deletions(self, group):
        scored = []
        for filepath in group:
            width, height, size_bytes = self._get_image_info(filepath)
            pixel_count = width * height
            scored.append((filepath, pixel_count, size_bytes))

        scored.sort(key=lambda entry: (entry[1], entry[2]), reverse=True)
        keeper = scored[0][0]
        to_delete = [filepath for filepath, _, _ in scored[1:]]
        return keeper, to_delete


    def _remove_exact_duplicates(self, images, test_run):
        print(f"Phase 1: Checking for exact duplicates among {len(images)} images...")
        self._log_group("Exactly Same Images:\n")

        hash_to_files = {}
        failed_files = []

        for index, filepath in enumerate(images, start=1):
            if index % 500 == 0:
                print(f"  Hashed {index}/{len(images)}")
            try:
                file_hash = self._compute_file_hash(filepath)
                hash_to_files.setdefault(file_hash, []).append(filepath)
            except Exception as error:
                print(f"  Warning: could not read {filepath}: {error}")
                failed_files.append(filepath)

        deleted_images = 0
        deleted_labels = 0
        removed_paths = set()

        duplicate_groups = {h: paths for h, paths in hash_to_files.items() if len(paths) > 1}

        for group_index, (_, group) in enumerate(duplicate_groups.items(), start=1):
            keeper, files_to_delete = self._pick_keeper_and_deletions(group)
            keeper_width, keeper_height, keeper_size = self._get_image_info(keeper)
            
            line = (
                f"\n\nExact Group {group_index}:\n"
                f"  Keeping: file://{os.path.join(self.root_directory, os.path.basename(keeper))}"
            )            
            self._log_group(line)

            for filepath in files_to_delete:
                width, height, size_bytes = self._get_image_info(filepath)
                action = "Would Delete:" if test_run else "Deleted:"
                
                line =(
                    f"  {action} file://{os.path.join(self.root_directory, os.path.basename(filepath))}"
                )
                self._log_group(line)

                if not test_run:
                    os.remove(filepath)
                    label_path = self._find_label_path(filepath)
                    if label_path:
                        os.remove(label_path)
                        deleted_labels += 1

                deleted_images += 1
                removed_paths.add(filepath)

        surviving_images = [path for path in images if path not in removed_paths]

        print(f"  Exact duplicates: {deleted_images} images, {deleted_labels} labels {'would be ' if test_run else ''}removed")
        print(f"  {len(surviving_images)} images remain for perceptual comparison\n")

        if failed_files:
            print(f"  {len(failed_files)} files could not be read:")
            for path in failed_files:
                print(f"    {path}")
            print()

        return surviving_images, deleted_images, deleted_labels


    def _find_perceptual_duplicate_groups(self, images, threshold):
        print(f"Phase 2: Computing perceptual hashes for {len(images)} images...")
        self._log_group("\n\nPerceptually Similar Images:")

        hashes = []
        failed_files = []

        for index, filepath in enumerate(images, start=1):
            if index % 500 == 0:
                print(f"  {index}/{len(images)}")
            try:
                perceptual_hash = self._compute_perceptual_hash(filepath)
                hashes.append((filepath, perceptual_hash))
            except Exception as error:
                print(f"  Warning: could not process {filepath}: {error}")
                failed_files.append(filepath)

        if failed_files:
            print(f"  {len(failed_files)} files could not be processed:")
            for path in failed_files:
                print(f"    {path}")
            print()

        print(f"Comparing {len(hashes)} hashes (this may take a while)...")

        parent_map = list(range(len(hashes)))

        def find_root(node):
            while parent_map[node] != node:
                parent_map[node] = parent_map[parent_map[node]]
                node = parent_map[node]
            return node

        def union(node_a, node_b):
            root_a = find_root(node_a)
            root_b = find_root(node_b)
            if root_a != root_b:
                parent_map[root_b] = root_a

        for i in range(len(hashes)):
            if (i + 1) % 1000 == 0:
                print(f"  Compared {i + 1}/{len(hashes)} anchors")
            for j in range(i + 1, len(hashes)):
                distance = hashes[i][1] - hashes[j][1]
                if distance <= threshold:
                    union(i, j)

        group_map = {}
        for index in range(len(hashes)):
            root = find_root(index)
            group_map.setdefault(root, []).append(hashes[index][0])

        groups = [members for members in group_map.values() if len(members) > 1]
        return groups


    def _delete_perceptual_duplicates(self, groups, test_run):
        total_deleted_images = 0
        total_deleted_labels = 0

        for group_index, group in enumerate(groups, start=1):
            keeper, files_to_delete = self._pick_keeper_and_deletions(group)
            keeper_width, keeper_height, keeper_size = self._get_image_info(keeper)
            
            line = (
                f"\n\nGroup {group_index}:\n"
                f"  Keeping: file://{os.path.join(self.root_directory, os.path.basename(keeper))}"
            )            
            self._log_group(line)

            for filepath in files_to_delete:
                width, height, size_bytes = self._get_image_info(filepath)
                action = "Would Delete:" if test_run else "Deleting:"
                
                line =(
                    f"  {action} file://{os.path.join(self.root_directory, os.path.basename(filepath))}"
                )
                self._log_group(line)
                
                if not test_run:
                    os.remove(filepath)
                    total_deleted_images += 1

                    label_path = self._find_label_path(filepath)
                    if label_path:
                        os.remove(label_path)
                        total_deleted_labels += 1
                else:
                    total_deleted_images += 1

        return total_deleted_images, total_deleted_labels


    def calculate_similarity(self):
        images = self._collect_images(self.root_directory)
        print(f"Found {len(images)} images in {self.root_directory}")
        print(f"pHash threshold: {PHASH_THRESHOLD}")
        print(f"Test run: {self.test_run}\n")

        if not images:
            sys.exit(0)

        surviving_images, exact_deleted_images, exact_deleted_labels = self._remove_exact_duplicates(images, self.test_run)

        if not surviving_images:
            print("No images left after exact dedup.")
            sys.exit(0)

        groups = self._find_perceptual_duplicate_groups(surviving_images, PHASH_THRESHOLD)

        if not groups:
            print("\nNo perceptual duplicates found.")
            print(f"\nTotal: {exact_deleted_images} images and {exact_deleted_labels} labels {'would be ' if self.test_run else ''}removed")
            sys.exit(0)

        perceptual_dupes = sum(len(group) - 1 for group in groups)
        print(f"\nFound {len(groups)} perceptual duplicate groups, {perceptual_dupes} files to {'delete' if not self.test_run else 'flag'}\n")

        perceptual_deleted_images, perceptual_deleted_labels = self._delete_perceptual_duplicates(groups, self.test_run)

        total_images = exact_deleted_images + perceptual_deleted_images
        total_labels = exact_deleted_labels + perceptual_deleted_labels
        action = "would be removed" if self.test_run else "removed"

        print(f"\nTotal: {total_images} images and {total_labels} labels {action}")
        print(f"  Exact duplicates: {exact_deleted_images}")
        print(f"  Perceptual duplicates: {perceptual_deleted_images}")