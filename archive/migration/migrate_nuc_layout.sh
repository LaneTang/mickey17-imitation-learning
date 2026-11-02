#!/usr/bin/env bash
# Move the temporary patch-4 runtime into the existing NUC IL directory.
# Preview by default; --apply moves files only after checking every conflict.
set -euo pipefail

staging="/home/team13/mikey17_patch4_preflight"
project="/home/team13/Projects/lerobot/atom's"
destination="$project/IL"
default_pose="$destination/default_pose.json"

if [[ ${1:-} != "" && ${1:-} != "--apply" ]]; then
  printf 'Usage: bash %s [--apply]\n' "$0" >&2
  exit 2
fi
if [[ ! -d "$staging" ]]; then
  printf 'Staging directory missing: %s\nNo files were moved.\n' "$staging" >&2
  exit 1
fi
if [[ -L "$destination" || ( -e "$destination" && ! -d "$destination" ) ]]; then
  printf 'Destination exists but is not a directory: %s\n' "$destination" >&2
  exit 1
fi
if [[ -n $(find "$staging" -type l -print -quit) ]]; then
  printf 'Staging contains a symbolic link; inspect it before migration.\n' >&2
  exit 1
fi
if [[ -n $(find "$staging" ! -type d ! -type f -print -quit) ]]; then
  printf 'Staging contains a special file; inspect it before migration.\n' >&2
  exit 1
fi

# Check all collisions BEFORE moving anything. Existing identical copies are
# safe to deduplicate; different contents must be resolved by the user.
files=0
while IFS= read -r -d '' source_file; do
  relative=${source_file#"$staging"/}
  target_file="$destination/$relative"
  if [[ -e "$target_file" || -L "$target_file" ]]; then
    if [[ -L "$target_file" || ! -f "$target_file" ]] || ! cmp -s -- "$source_file" "$target_file"; then
      printf 'CONFLICT: %s\n  differs from %s\nNo files were moved.\n' \
        "$source_file" "$target_file" >&2
      exit 1
    fi
  fi
  ((files += 1))
done < <(find "$staging" -type f -print0)

printf 'Source:      %s\nDestination: %s\n' "$staging" "$destination"
printf 'Files checked: %d; no conflicting destinations.\n' "$files"
if [[ -f "$default_pose" ]]; then
  printf 'Existing default pose stays at: %s\n' "$default_pose"
else
  printf 'Default pose missing: %s (restore it before live preflight).\n' "$default_pose"
fi
if [[ ${1:-} != "--apply" ]]; then
  printf 'PREVIEW ONLY. Run again with --apply to move these files.\n'
  exit 0
fi

mkdir -p -- "$destination"
while IFS= read -r -d '' source_dir; do
  relative=${source_dir#"$staging"/}
  if [[ "$source_dir" != "$staging" ]]; then
    mkdir -p -- "$destination/$relative"
  fi
done < <(find "$staging" -type d -print0)

moved=0
deduplicated=0
while IFS= read -r -d '' source_file; do
  relative=${source_file#"$staging"/}
  target_file="$destination/$relative"
  if [[ -e "$target_file" ]]; then
    # The preflight cmp above established byte-for-byte identity.
    rm -- "$source_file"
    ((deduplicated += 1))
  else
    mv -- "$source_file" "$target_file"
    ((moved += 1))
  fi
done < <(find "$staging" -type f -print0)
find "$staging" -depth -type d -empty -delete
if [[ -e "$staging" ]]; then
  printf 'Some staging entries remain; inspect %s\n' "$staging" >&2
  exit 1
fi

printf 'MIGRATED: %d files moved, %d identical duplicates removed.\n' "$moved" "$deduplicated"
printf 'Canonical runtime: %s\n' "$destination"
printf 'Existing default pose was not modified: %s\n' "$default_pose"
